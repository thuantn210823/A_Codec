import glob
import logging
import math
import os
import random
import sys
import warnings

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torchaudio.transforms import MelSpectrogram

# ============================================================
# EnCodec reference code (docs/encodec-main). On Kaggle: upload the
# `encodec-main` folder as a dataset and point ENCODEC_ROOT at it.
# Requires `einops`.
# ============================================================
ENCODEC_ROOT = os.environ.get(
    "ENCODEC_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "encodec-main"),
)
if ENCODEC_ROOT not in sys.path:
    sys.path.insert(0, ENCODEC_ROOT)

from encodec.model import EncodecModel                      # noqa: E402
from encodec.msstftd import MultiScaleSTFTDiscriminator     # noqa: E402
from encodec.distrib import average_metrics                 # noqa: E402
from encodec.quantization import core_vq                    # noqa: E402

# core_vq warns on *every* training forward about issue #25 -> silence it
warnings.filterwarnings("ignore", message="When using RVQ in training model")
# n_fft=32 with 64 mel bins yields some empty filters (harmless)
warnings.filterwarnings("ignore", message=".*empty filterbank.*")


# ============================================================
# RVQ fix for https://github.com/facebookresearch/encodec/issues/25
# Original: residual = residual - quantized, where `quantized` carries the
# straight-through estimator -> the residual fed to codebook k>1 has zero
# gradient w.r.t. the encoder output, so later commitment losses never reach
# the encoder.
# Fix (same as audiocraft): detach every layer's output so each residual is
# x - sum(detached q_j) (commitment losses of all layers reach the encoder),
# then apply ONE straight-through estimator on the sum. Detaching only the
# residual (but summing the per-layer STE outputs) would multiply the decoder
# gradient into the encoder by n_q (up to 32x) -> exploding commitment loss.
# ============================================================
FIX_RVQ_ISSUE_25 = True


def _rvq_forward_fixed(self, x, n_q=None):
    quantized_out = 0.0
    residual = x
    all_losses, all_indices = [], []
    n_q = n_q or len(self.layers)
    for layer in self.layers[:n_q]:
        quantized, indices, loss = layer(residual)
        quantized = quantized.detach()
        residual = residual - quantized
        quantized_out = quantized_out + quantized
        all_indices.append(indices)
        all_losses.append(loss)
    if self.training:
        quantized_out = x + (quantized_out - x).detach()      # single STE: d out / d x = I
    out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
    return quantized_out, out_indices, out_losses


if FIX_RVQ_ISSUE_25:
    core_vq.ResidualVectorQuantization.forward = _rvq_forward_fixed


# ============================================================
# Data
# ============================================================
class LibriTTS_Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, sample_rate: int = 24000, segment: float = 1.0, random_crop: bool = True):
        super().__init__()
        self.root = root
        self.data = sorted(glob.glob(f'{root}/*/*/*.wav'))
        self.sample_rate = sample_rate
        self.segment_length = int(segment * sample_rate)
        self.random_crop = random_crop

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.data[idx])
        wav = wav.mean(dim=0, keepdim=True)                     # mono [1, T]
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        T, L_seg = wav.shape[-1], self.segment_length
        if T > L_seg:
            start = random.randint(0, T - L_seg) if self.random_crop else 0
            wav = wav[:, start:start + L_seg]
        elif T < L_seg:
            wav = F.pad(wav, (0, L_seg - T))
        return wav

    def __len__(self):
        return len(self.data)


class LibriTTS(L.LightningDataModule):
    def __init__(self, batch_size: int, sample_rate: int = 24000, segment: float = 1.0):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = LibriTTS_Dataset('/kaggle/input/datasets/pratt3000/libritts/LibriTTS/train-clean-100',
                                              sample_rate, segment, random_crop=True)
        self.val_dataset = LibriTTS_Dataset('/kaggle/input/datasets/luizfelipebjcosta/libri-tts-dev/dev-clean/dev-clean',
                                            sample_rate, segment, random_crop=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=True,
                                           collate_fn=self.collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=2,
                                           collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch, dim=0)                       # [B, 1, T]


# ============================================================
# Loss functions (EnCodec paper, Sec. 3.4)
# ============================================================
def time_loss(x, G_x):
    # l_t: L1 on the waveform
    return F.l1_loss(G_x, x)


def freq_loss(x, G_x, melspecs):
    # l_f: L1 + L2 over 64-bin mel spectrograms, scales 2^5 .. 2^11, hop = win/4
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, G_x = x.float(), G_x.float()
        total = 0.0
        for melspec in melspecs:
            S_x, S_G_x = melspec(x), melspec(G_x)
            total = total + F.l1_loss(S_G_x, S_x) + F.mse_loss(S_G_x, S_x)
    return total / len(melspecs)


def adversarial_g_loss(logits_G_x):
    # l_g: mean_k max(0, 1 - D_k(G(x)))
    return torch.stack([F.relu(1 - l).mean() for l in logits_G_x]).mean()


def feature_loss(fmaps_x, fmaps_G_x, eps=1e-8):
    # l_feat: mean_{k,l} ||D_k^l(x) - D_k^l(G(x))||_1 / mean(||D_k^l(x)||_1)
    losses = []
    for fmap_x, fmap_G_x in zip(fmaps_x, fmaps_G_x):
        for f_x, f_G_x in zip(fmap_x, fmap_G_x):
            losses.append((f_x - f_G_x).abs().mean() / (f_x.abs().mean() + eps))
    return torch.stack(losses).mean()


def adversarial_d_loss(logits_x, logits_G_x):
    # hinge: mean_k [max(0, 1 - D_k(x)) + max(0, 1 + D_k(G(x)))]
    return torch.stack([
        F.relu(1 - l_x).mean() + F.relu(1 + l_G_x).mean()
        for l_x, l_G_x in zip(logits_x, logits_G_x)
    ]).mean()


# ============================================================
# Loss balancer (same math as encodec/balancer.py) but it only *returns* the
# combined gradient w.r.t. G_x instead of calling G_x.backward() itself, so the
# generator update can go through a single `manual_backward` (DDP/AMP-safe).
# ============================================================
class Balancer:
    def __init__(self, weights, rescale_grads=True, total_norm=1.0, ema_decay=0.999,
                 per_batch_item=True, epsilon=1e-12):
        self.weights = weights
        self.rescale_grads = rescale_grads
        self.total_norm = total_norm
        self.ema_decay = ema_decay
        self.per_batch_item = per_batch_item
        self.epsilon = epsilon
        self._total = {}
        self._fix = {}

    def _ema(self, metrics):
        for k, v in metrics.items():
            self._total[k] = self._total.get(k, 0.0) * self.ema_decay + float(v)
            self._fix[k] = self._fix.get(k, 0.0) * self.ema_decay + 1.0
        return {k: self._total[k] / self._fix[k] for k in metrics}

    def compute_grad(self, losses, G_x):
        norms, grads = {}, {}
        for name, loss in losses.items():
            grad, = autograd.grad(loss, [G_x], retain_graph=True)
            if self.per_batch_item:
                norm = grad.norm(dim=tuple(range(1, grad.dim()))).mean()
            else:
                norm = grad.norm()
            norms[name] = norm.detach()
            grads[name] = grad
        count = G_x.shape[0] if self.per_batch_item else 1
        avg_norms = average_metrics(self._ema(norms), count)    # all-reduced across ranks

        total_weights = sum(self.weights[k] for k in avg_norms)
        out_grad = 0
        for name, avg_norm in avg_norms.items():
            if self.rescale_grads:
                scale = (self.weights[name] / total_weights) * self.total_norm / (self.epsilon + avg_norm)
                out_grad = out_grad + grads[name] * scale
            else:
                out_grad = out_grad + grads[name] * self.weights[name]
        ratios = {f"ratio_{k}": v / (sum(avg_norms.values()) + self.epsilon) for k, v in avg_norms.items()}
        return out_grad.detach(), ratios


# ============================================================
# Lightning module
# ============================================================
class EncodecTask(L.LightningModule):
    def __init__(self,
        encodec: EncodecModel,
        disc: nn.Module,
        sample_rate: int = 24000,
        lambda_t: float = 0.1,
        lambda_f: float = 1.0,
        lambda_g: float = 3.0,
        lambda_feat: float = 3.0,
        lambda_w: float = 1.0,
        use_balancer: bool = True,
        disc_update_prob: float = 2 / 3,
        eval_bandwidth: float = 6.0,
        lr: float = 3e-4,
        betas=(0.5, 0.9),
        accumulate_grad_batches: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encodec", "disc"])

        # GAN = 2 optimizers stepping alternately -> manual optimization
        self.automatic_optimization = False

        self.encodec = encodec
        self.disc = disc

        self.melspecs = nn.ModuleList([
            MelSpectrogram(sample_rate=sample_rate, n_fft=2 ** i, win_length=2 ** i,
                           hop_length=2 ** i // 4, n_mels=64, power=1.0)
            for i in range(5, 12)
        ])
        self.balancer = Balancer(
            weights={"t": lambda_t, "f": lambda_f, "g": lambda_g, "feat": lambda_feat},
            rescale_grads=use_balancer,
        )

    # ---------------- helpers ----------------
    def _generate(self, x, bandwidth):
        # Training-mode forward: encoder -> RVQ (straight-through + commitment) -> decoder.
        # (EncodecModel.forward uses quantizer.encode/decode, which is not trainable.)
        emb = self.encodec.encoder(x)
        self._emb_rms = emb.detach().pow(2).mean().sqrt()     # latent-scale diagnostic
        q_res = self.encodec.quantizer(emb, self.encodec.frame_rate, bandwidth)
        G_x = self.encodec.decoder(q_res.quantized)
        T = x.shape[-1]
        if G_x.shape[-1] > T:
            G_x = G_x[..., :T]
        elif G_x.shape[-1] < T:
            G_x = F.pad(G_x, (0, T - G_x.shape[-1]))
        return G_x, q_res.penalty

    def _rng(self, *keys):
        # Same seed on every DDP rank -> same bandwidth / D-update decision everywhere
        # (k-means init and dead-code expiry call torch.distributed.broadcast).
        seed = 0
        for k in keys:
            seed = seed * 1_000_003 + int(k)
        return random.Random(seed)

    def _sample_bandwidth(self, batch_idx):
        if self.current_epoch == 0 and batch_idx == 0:
            return max(self.encodec.target_bandwidths)       # init every codebook with k-means
        return self._rng(self.current_epoch, batch_idx, 1).choice(self.encodec.target_bandwidths)

    def _generator_losses(self, x, G_x):
        logits_G_x, fmaps_G_x = self.disc(G_x)
        with torch.no_grad():
            _, fmaps_x = self.disc(x)
        return {
            "t": time_loss(x, G_x),
            "f": freq_loss(x, G_x, self.melspecs),
            "g": adversarial_g_loss(logits_G_x),
            "feat": feature_loss(fmaps_x, fmaps_G_x),
        }

    def _weighted(self, losses):
        return sum(self.balancer.weights[k] * v for k, v in losses.items())

    def _discriminator_loss(self, x, G_x):
        logits_x, _ = self.disc(x)
        logits_G_x, _ = self.disc(G_x.detach())
        return adversarial_d_loss(logits_x, logits_G_x)

    # ---------------- train ----------------
    def training_step(self, batch, batch_idx):
        x = batch
        opt_g, opt_d = self.optimizers()
        hp = self.hparams

        acc = hp.accumulate_grad_batches
        is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
        do_step = ((batch_idx + 1) % acc == 0) or is_last_batch
        bandwidth = self._sample_bandwidth(batch_idx)
        # one D-update decision per accumulation group
        update_d = self._rng(self.current_epoch, batch_idx // acc, 2).random() < hp.disc_update_prob

        # ----- Generator step -----
        self.toggle_optimizer(opt_g)                 # freezes D params during G backward
        if (batch_idx % acc) == 0:
            opt_g.zero_grad()
        G_x, commit_loss = self._generate(x, bandwidth)
        losses = self._generator_losses(x, G_x)
        out_grad, ratios = self.balancer.compute_grad(losses, G_x)
        # surrogate whose gradient w.r.t. G_x is exactly the balanced gradient
        surrogate = (G_x * out_grad).sum() + hp.lambda_w * commit_loss
        self.manual_backward(surrogate / acc)
        if do_step:
            opt_g.step()
        self.untoggle_optimizer(opt_g)

        # ----- Discriminator step (with prob. disc_update_prob) -----
        if update_d:
            self.toggle_optimizer(opt_d)             # freezes G params during D backward
            if (batch_idx % acc) == 0:
                opt_d.zero_grad()
            loss_d = self._discriminator_loss(x, G_x)
            self.manual_backward(loss_d / acc)
            if do_step:
                opt_d.step()
            self.untoggle_optimizer(opt_d)
            self.log("tld", loss_d, on_step=True, on_epoch=True, sync_dist=True)

        loss_g = self._weighted({k: v.detach() for k, v in losses.items()}) + hp.lambda_w * commit_loss.detach()
        self.log("tlg", loss_g, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({"tt": losses["t"], "tf": losses["f"], "tg": losses["g"],
                       "tfeat": losses["feat"], "tc": commit_loss},
                      prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("bw", float(bandwidth), on_step=True, on_epoch=False)
        self.log("emb_rms", self._emb_rms, on_step=True, on_epoch=False)
        self.log_dict(ratios, on_step=True, on_epoch=False)

    # ---------------- val / test ----------------
    def _eval_step(self, batch, prefix):
        x = batch
        G_x, commit_loss = self._generate(x, self.hparams.eval_bandwidth)
        losses = self._generator_losses(x, G_x)
        loss_g = self._weighted(losses) + self.hparams.lambda_w * commit_loss
        loss_d = self._discriminator_loss(x, G_x)
        loss_rec = losses["t"] + losses["f"]

        self.log(f"{prefix}_loss_g", loss_g, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_loss_d", loss_d, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_loss_rec", loss_rec, on_epoch=True, sync_dist=True)
        return loss_g

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    # ---------------- optimizers ----------------
    def configure_optimizers(self):
        hp = self.hparams
        optimizer_g = torch.optim.Adam(self.encodec.parameters(), lr=hp.lr, betas=hp.betas)
        optimizer_d = torch.optim.Adam(self.disc.parameters(), lr=hp.lr, betas=hp.betas)
        logging.info(f"Estimated stepping batches: {self.trainer.estimated_stepping_batches}")
        return [optimizer_g, optimizer_d]


# ============================================================
# Model builders
# ============================================================
TARGET_BANDWIDTHS = [1.5, 3., 6., 12., 24.]
SAMPLE_RATE = 24000


def build_encodec():
    # Same config as EncodecModel.encodec_model_24khz (causal, weight_norm, n_q=32, bins=1024)
    return EncodecModel._get_model(TARGET_BANDWIDTHS, SAMPLE_RATE, channels=1,
                                   causal=True, model_norm='weight_norm',
                                   audio_normalize=False, name='unset')


def build_discriminator():
    # MS-STFT discriminator, 5 scales as in the paper
    n_ffts = [2048, 1024, 512, 256, 128]
    return MultiScaleSTFTDiscriminator(filters=32, n_ffts=n_ffts,
                                       hop_lengths=[n // 4 for n in n_ffts],
                                       win_lengths=n_ffts)


# ============================================================
# Progress bar (from your template)
# ============================================================
class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        return bar


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(dirpath='/kaggle/working/checkpoint_encodec',
                                          save_top_k=-1, monitor='val_loss_rec',
                                          mode='min', save_last=True,
                                          filename='EnCodec-{epoch}-{val_loss_rec:.6f}')
    logger = TensorBoardLogger("/kaggle/working/log", name="EnCodec", version='version_0')

    L.seed_everything(42, workers=True)

    LAMBDA_T = 0.1
    LAMBDA_F = 1
    LAMBDA_G = 3
    LAMBDA_FEAT = 3
    LAMBDA_W = 1
    N_EPOCHS = 7
    BATCH_SIZE = 8
    SEGMENT = 1.0          # seconds per training example (paper: 1 s)

    encodec = build_encodec()
    disc = build_discriminator()

    model = EncodecTask(
        encodec=encodec,
        disc=disc,
        sample_rate=SAMPLE_RATE,
        lambda_t=LAMBDA_T,
        lambda_f=LAMBDA_F,
        lambda_g=LAMBDA_G,
        lambda_feat=LAMBDA_FEAT,
        lambda_w=LAMBDA_W,
        use_balancer=True,
        disc_update_prob=2 / 3,
        eval_bandwidth=6.0,
        lr=3e-4,
        betas=(0.5, 0.9),
        accumulate_grad_batches=2,   # handled manually (Trainer arg not allowed with manual optimization)
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    data = LibriTTS(BATCH_SIZE, sample_rate=SAMPLE_RATE, segment=SEGMENT)

    trainer = L.Trainer(accelerator="gpu",
                        callbacks=[checkpoint_callback, CustomProgressBar()],
                        enable_checkpointing=True,
                        max_epochs=N_EPOCHS,
                        # precision='16-mixed',       # use '32-true' if the GAN becomes unstable
                        logger=logger,
                        strategy='ddp_find_unused_parameters_true',
                        devices=[0, 1],
                        num_sanity_val_steps=0,       # don't let val data k-means-init the codebooks
                        deterministic=False)

    trainer.fit(model, data)

    # Plain EncodecModel weights -> usable with encodec's encode/decode/compress
    if trainer.is_global_zero:
        torch.save(model.encodec.state_dict(), '/kaggle/working/encodec_24khz_trained.th')
