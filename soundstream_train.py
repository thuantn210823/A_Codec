import glob
import os
from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pytorch_lightning as L

from vector_quantize_pytorch import ResidualVQ

class LibriTTS_Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        self.data = glob.glob(f'{root}/*/*/*.wav')

    def __getitem__(self, idx):
        audio_filepath = self.data[idx]
        wav, sr = torchaudio.load(audio_filepath)
        return wav[:, :sr*4], sr
        
    def __len__(self):
        return len(self.data)

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels=out_channels//2, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=2*out_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),

        )

    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            nn.ELU(),
            EncoderBlock(out_channels=2*C, stride=2),
            nn.ELU(),
            EncoderBlock(out_channels=4*C, stride=4),
            nn.ELU(),
            EncoderBlock(out_channels=8*C, stride=5),
            nn.ELU(),
            EncoderBlock(out_channels=16*C, stride=8),
            nn.ELU(),
            CausalConv1d(in_channels=16*C, out_channels=D, kernel_size=3)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=D, out_channels=16*C, kernel_size=7),
            nn.ELU(),
            DecoderBlock(out_channels=8*C, stride=8),
            nn.ELU(),
            DecoderBlock(out_channels=4*C, stride=5),
            nn.ELU(),
            DecoderBlock(out_channels=2*C, stride=4),
            nn.ELU(),
            DecoderBlock(out_channels=C, stride=2),
            nn.ELU(),
            CausalConv1d(in_channels=C, out_channels=1, kernel_size=7)
        )
    
    def forward(self, x):
        return self.layers(x)

class SoundStream(nn.Module):
    def __init__(self, C, D, n_q, codebook_size):
        super().__init__()

        self.encoder = Encoder(C=C, D=D)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q, dim=D, codebook_size=codebook_size,
            kmeans_init=True, 
            kmeans_iters=100, 
            dead_code_threshold_ratio=0.1, 
            decay = 0.99, 
            quantize_dropout = True,
            quantize_dropout_cutoff_index = 0
        )
        self.decoder = Decoder(C=C, D=D)
    
    def forward(self, x):
        e = self.encoder(x).transpose(1, 2)
        quantized, indices, commit_loss = self.quantizer(e)
        o = self.decoder(quantized.transpose(1, 2))
        return o, commit_loss.sum()

# Wave-based Discriminator


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class WaveDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                WNConv1d(in_channels=1, out_channels=16, kernel_size=15),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=16, out_channels=64, kernel_size=41,
                         stride=4, padding=20, groups=4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=64, out_channels=256, kernel_size=41,
                         stride=4, padding=20, groups=16),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=256, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=64),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=256),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=5,
                         stride=1, padding=2),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            WNConv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1,
                     padding=1)
        ])
    
    def features_lengths(self, lengths):
        return [
            lengths,
            torch.div(lengths+3, 4, rounding_mode="floor"),
            torch.div(lengths+15, 16, rounding_mode="floor"),
            torch.div(lengths+63, 64, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map


class WaveDiscriminator(nn.Module):
    def __init__(self, num_D, downsampling_factor):
        super().__init__()
        
        self.num_D = num_D
        self.downsampling_factor = downsampling_factor
        
        self.model = nn.ModuleDict({
            f"disc_{downsampling_factor**i}": WaveDiscriminatorBlock()
            for i in range(num_D)
        })
        self.downsampler = nn.AvgPool1d(kernel_size=4, stride=2, padding=1,
                                        count_include_pad=False)
    
    def features_lengths(self, lengths):
        return {
            f"disc_{self.downsampling_factor**i}": self.model[f"disc_{self.downsampling_factor**i}"].features_lengths(torch.div(lengths, 2**i, rounding_mode="floor")) for i in range(self.num_D)
        }
    
    def forward(self, x):
        results = {}
        for i in range(self.num_D):
            disc = self.model[f"disc_{self.downsampling_factor**i}"]
            results[f"disc_{self.downsampling_factor**i}"] = disc(x)
            x = self.downsampler(x)
        return results


# STFT-based Discriminator

class ResidualUnit2d(nn.Module):
    def __init__(self, in_channels, N, m, s_t, s_f):
        super().__init__()
        
        self.s_t = s_t
        self.s_f = s_f

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=N,
                kernel_size=(3, 3),
                padding="same"
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=N,
                out_channels=m*N,
                kernel_size=(s_f+2, s_t+2),
                stride=(s_f, s_t)
            )
        )
        
        self.skip_connection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=m*N,
            kernel_size=(1, 1), stride=(s_f, s_t)
        )

    def forward(self, x):
        return self.layers(F.pad(x, [self.s_t+1, 0, self.s_f+1, 0])) + self.skip_connection(x)


class STFTDiscriminator(nn.Module):
    def __init__(self, C, F_bins):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(7, 7)),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=32,  N=C,   m=2, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=2*C, N=2*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*C, N=4*C, m=1, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*C, N=4*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*C, N=8*C, m=1, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*C,  N=8*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Conv2d(in_channels=16*C, out_channels=1,
                      kernel_size=(F_bins//2**6, 1))
        ])
    
    def features_lengths(self, lengths):
        return [
            lengths-6,
            lengths-6,
            torch.div(lengths-5, 2, rounding_mode="floor"),
            torch.div(lengths-5, 2, rounding_mode="floor"),
            torch.div(lengths-3, 4, rounding_mode="floor"),
            torch.div(lengths-3, 4, rounding_mode="floor"),
            torch.div(lengths+1, 8, rounding_mode="floor"),
            torch.div(lengths+1, 8, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map

class LibriTTS(L.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = LibriTTS_Dataset('/kaggle/input/datasets/pratt3000/libritts/LibriTTS/train-clean-100')
        self.val_dataset = LibriTTS_Dataset('/kaggle/input/datasets/luizfelipebjcosta/libri-tts-dev/dev-clean/dev-clean')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = True,
                                           num_workers = 2,
                                           collate_fn = self.collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = False,
                                           num_workers = 2,
                                           collate_fn = self.collate_fn)

    def collate_fn(self, batch):
        lengths = torch.tensor([elem[0].shape[-1] for elem in batch])
        wavs = [elem[0].transpose(0, 1).contiguous() for elem in batch]
        return nn.utils.rnn.pad_sequence(wavs, batch_first=True).transpose(1, 2).contiguous(), lengths

import logging
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torchaudio.transforms import MelSpectrogram


# ============================================================
# Loss functions (same math as your original code)
# ============================================================
def adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    stft_loss = F.relu(1 - features_stft_disc_G_x[-1]).sum(dim=3).flatten() / lengths_stft[-1].flatten()
    wave_loss = torch.cat([
        F.relu(1 - features_wave_disc_G_x[key][-1]).sum(dim=2).flatten() / lengths_wave[key][-1].flatten()
        for key in lengths_wave.keys()
    ])
    return torch.cat([stft_loss, wave_loss]).mean()


def feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x,
                 lengths_wave, lengths_stft):
    stft_losses = []
    for i, (feat_x, feat_G_x) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x)):
        valid_t = lengths_stft[i].view(-1, 1, 1, 1)
        l1 = (feat_x - feat_G_x).abs().sum(dim=-1, keepdim=True) / valid_t
        norm = feat_x.abs().sum(dim=-1, keepdim=True) / valid_t + 1e-8
        stft_losses.append((l1 / norm).mean())

    wave_losses = []
    for key in lengths_wave.keys():
        for i, (feat_x, feat_G_x) in enumerate(zip(features_wave_disc_x[key], features_wave_disc_G_x[key])):
            valid_t = lengths_wave[key][i].view(-1, 1, 1)
            l1 = (feat_x - feat_G_x).abs().sum(dim=-1, keepdim=True) / valid_t
            norm = feat_x.abs().sum(dim=-1, keepdim=True) / valid_t + 1e-8
            wave_losses.append((l1 / norm).mean())

    return (torch.stack(stft_losses).mean() + torch.stack(wave_losses).mean()) * 0.5


def spectral_reconstruction_loss(x, G_x, melspecs, eps=1e-5):
    # Eq. (4)-(5): s = 2^6 ... 2^11, hop = s/4, 64 mel bin, alpha_s = sqrt(s/2)
    # Dùng MEAN thay vì SUM: sum trên (batch x 64 mel x hàng nghìn frame x 6 scale)
    # cho loss ~1e6, lấn át hoàn toàn adv/feat và làm tràn fp16 khi GradScaler nhân loss.
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, G_x = x.float(), G_x.float()
        L_total = 0.0
        for melspec in melspecs:
            s = melspec.n_fft
            alpha_s = (s / 2) ** 0.5
            S_x = melspec(x)            # [B, 1, n_mels, frames]
            S_G_x = melspec(G_x)
            l1 = (S_x - S_G_x).abs().mean()
            log_l2 = ((torch.log(S_x + eps) - torch.log(S_G_x + eps)) ** 2).sum(dim=-2).sqrt().mean()
            L_total = L_total + l1 + alpha_s * log_l2
    return L_total


def adversarial_d_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x,
                       lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()

    real_stft_loss = F.relu(1 - features_stft_disc_x[-1]).sum(dim=3).squeeze() / lengths_stft[-1].squeeze()
    real_wave_loss = torch.stack([
        F.relu(1 - features_wave_disc_x[key][-1]).sum(dim=-1).squeeze() / lengths_wave[key][-1].squeeze()
        for key in wave_disc_names
    ], dim=1)
    real_loss = torch.cat([real_stft_loss.view(-1, 1), real_wave_loss], dim=1).mean()

    generated_stft_loss = F.relu(1 + features_stft_disc_G_x[-1]).sum(dim=-1).squeeze() / lengths_stft[-1].squeeze()
    generated_wave_loss = torch.stack([
        F.relu(1 + features_wave_disc_G_x[key][-1]).sum(dim=-1).squeeze() / lengths_wave[key][-1].squeeze()
        for key in wave_disc_names
    ], dim=1)
    generated_loss = torch.cat([generated_stft_loss.view(-1, 1), generated_wave_loss], dim=1).mean()

    return real_loss + generated_loss


# ============================================================
# Lightning module
# ============================================================
class SoundStreamTask(L.LightningModule):
    def __init__(self,
        soundstream: nn.Module,
        wave_disc: nn.Module,
        stft_disc: nn.Module,
        sample_rate: int = 16000,
        lambda_adv: float = 1.0,
        lambda_feat: float = 100.0,
        lambda_rec: float = 1.0,
        lr: float = 1e-4,
        betas=(0.5, 0.9),
        accumulate_grad_batches: int = 1,
    ):
        super(SoundStreamTask, self).__init__()
        # Modules are passed in as objects, so keep them out of hparams
        self.save_hyperparameters(ignore=["soundstream", "wave_disc", "stft_disc"])

        # GAN = 2 optimizers stepping alternately -> manual optimization
        self.automatic_optimization = False

        self.soundstream = soundstream
        self.wave_disc = wave_disc
        self.stft_disc = stft_disc

        # Build mel transforms once (the original re-created them every call)
        self.melspecs = nn.ModuleList([
            MelSpectrogram(sample_rate=sample_rate, n_fft=2 ** i, hop_length=2 ** i // 4, n_mels=64, power=1.0)
            for i in range(6, 12)
        ])
        self.register_buffer("stft_window", torch.hann_window(1024), persistent=False)

    # ---------------- helpers ----------------
    def _stft(self, wav):
        # wav: [B, 1, T] -> [B, 2, F, frames]  (same layout as return_complex=False + permute)
        spec = torch.stft(wav.squeeze(1).float(), n_fft=1024, hop_length=256,
                          window=self.stft_window, return_complex=True)
        return torch.view_as_real(spec).permute(0, 3, 1, 2)

    def _generate(self, x):
        # SoundStream downsamples/upsamples by its total stride (e.g. 2*4*5*8 = 320),
        # so G_x can come out a few samples longer/shorter than x when T is not a
        # multiple of that stride. Crop or zero-pad G_x to exactly x's length.
        G_x, commit_loss = self.soundstream(x)
        T = x.shape[-1]
        if G_x.shape[-1] > T:
            G_x = G_x[..., :T]
        elif G_x.shape[-1] < T:
            G_x = F.pad(G_x, (0, T - G_x.shape[-1]))
        return G_x, commit_loss

    def _lengths(self, lengths_x):
        lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
        lengths_stft = self.stft_disc.features_lengths(lengths_s_x)
        lengths_wave = self.wave_disc.features_lengths(lengths_x)
        return lengths_stft, lengths_wave

    def _generator_loss(self, x, G_x, s_x, s_G_x, lengths_stft, lengths_wave, commit_loss):
        features_stft_disc_x = self.stft_disc(s_x)
        features_wave_disc_x = self.wave_disc(x)
        features_stft_disc_G_x = self.stft_disc(s_G_x)
        features_wave_disc_G_x = self.wave_disc(G_x)

        hp = self.hparams
        loss_adv = adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave)
        loss_feat = feature_loss(features_stft_disc_x, features_wave_disc_x,
                                 features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
        loss_rec = spectral_reconstruction_loss(x, G_x, self.melspecs)
        loss_g = hp.lambda_adv * loss_adv + hp.lambda_feat * loss_feat + hp.lambda_rec * loss_rec + commit_loss
        return loss_g, loss_adv, loss_feat, loss_rec

    def _discriminator_loss(self, x, G_x, s_x, s_G_x, lengths_stft, lengths_wave):
        features_stft_disc_x = self.stft_disc(s_x)
        features_wave_disc_x = self.wave_disc(x)
        features_stft_disc_G_x_det = self.stft_disc(s_G_x.detach())
        features_wave_disc_G_x_det = self.wave_disc(G_x.detach())
        return adversarial_d_loss(features_stft_disc_x, features_wave_disc_x,
                                  features_stft_disc_G_x_det, features_wave_disc_G_x_det,
                                  lengths_stft, lengths_wave)

    # ---------------- train ----------------
    def training_step(self, batch, batch_idx):
        x, lengths_x = batch
        opt_g, opt_d = self.optimizers()

        acc = self.hparams.accumulate_grad_batches
        is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
        do_step = ((batch_idx + 1) % acc == 0) or is_last_batch

        lengths_stft, lengths_wave = self._lengths(lengths_x)
        s_x = self._stft(x)

        # ----- Generator step -----
        self.toggle_optimizer(opt_g)                 # freezes D params during G backward
        if (batch_idx % acc) == 0:
            opt_g.zero_grad()
        G_x, commit_loss = self._generate(x)
        s_G_x = self._stft(G_x)
        loss_g, loss_adv, loss_feat, loss_rec = self._generator_loss(
            x, G_x, s_x, s_G_x, lengths_stft, lengths_wave, commit_loss)
        self.manual_backward(loss_g / acc)
        if do_step:
            opt_g.step()
        self.untoggle_optimizer(opt_g)
        
        # ----- Discriminator step -----
        self.toggle_optimizer(opt_d)                 # freezes G params during D backward
        if (batch_idx % acc) == 0:
            opt_d.zero_grad()
        loss_d = self._discriminator_loss(x, G_x, s_x, s_G_x, lengths_stft, lengths_wave)
        self.manual_backward(loss_d / acc)
        if do_step:
            opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.log("tlg", loss_g, on_step=True, on_epoch=True, sync_dist=True)
        self.log("tld", loss_d, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({"ta": loss_adv, "tr": loss_rec, "tf": loss_feat, "tc": commit_loss},
                      prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

    # ---------------- val / test ----------------
    def _eval_step(self, batch, prefix):
        x, lengths_x = batch
        lengths_stft, lengths_wave = self._lengths(lengths_x)

        G_x, commit_loss = self._generate(x)
        s_x = self._stft(x)
        s_G_x = self._stft(G_x)

        loss_g, _, _, loss_rec = self._generator_loss(
            x, G_x, s_x, s_G_x, lengths_stft, lengths_wave, commit_loss)
        loss_d = self._discriminator_loss(x, G_x, s_x, s_G_x, lengths_stft, lengths_wave)

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
        optimizer_g = torch.optim.Adam(self.soundstream.parameters(), lr=hp.lr, betas=hp.betas)
        optimizer_d = torch.optim.Adam(
            list(self.wave_disc.parameters()) + list(self.stft_disc.parameters()),
            lr=hp.lr, betas=hp.betas)
        logging.info(f"Estimated stepping batches: {self.trainer.estimated_stepping_batches}")
        return [optimizer_g, optimizer_d]


# ============================================================
# Callbacks / logger / progress bar (from your template)
# ============================================================
checkpoint_callback = ModelCheckpoint(dirpath='/kaggle/working/checkpoint_soundstream',
                                      save_top_k=-1, monitor='val_loss_rec',
                                      mode='min', save_last=True,
                                      filename='SoundStream-{epoch}-{val_loss_rec:.6f}')
logger = TensorBoardLogger("/kaggle/working/log", name="SoundStream", version='version_0')


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

L.seed_everything(42, workers=True)


LAMBDA_ADV = 1
LAMBDA_FEAT = 100
LAMBDA_REC = 1
N_EPOCHS = 7
BATCH_SIZE = 2

# Build these exactly as in your original script
soundstream = SoundStream(C=32, D=128, n_q=8, codebook_size=1024)
wave_disc   = WaveDiscriminator(num_D=3, downsampling_factor=2)
W, H = 1024, 256
stft_disc   = STFTDiscriminator(C=32, F_bins=W//2)

model = SoundStreamTask(
    soundstream=soundstream,
    wave_disc=wave_disc,
    stft_disc=stft_disc,
    sample_rate=24000,
    lambda_adv=LAMBDA_ADV,
    lambda_feat=LAMBDA_FEAT,
    lambda_rec=LAMBDA_REC,
    lr=1e-4,
    betas=(0.5, 0.9),
    accumulate_grad_batches=2,   # handled manually (Trainer arg not allowed with manual optimization)
)

total_params = sum(p.numel() for p in model.parameters())
print(total_params)

data = LibriTTS(BATCH_SIZE)

trainer = L.Trainer(accelerator="gpu",
                    callbacks=[checkpoint_callback, CustomProgressBar()],
                    enable_checkpointing=True,
                    max_epochs=N_EPOCHS,
                    # precision='16-mixed',       # use '32-true' if the GAN becomes unstable
                    logger=logger,
                    strategy='ddp_find_unused_parameters_true',
                    devices=[0, 1],
                    deterministic=False)        # torch.stft / some conv backward kernels aren't deterministic

trainer.fit(model, data)