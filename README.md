# A_Codec

Neural audio codec training code for the LastDiar project, built on PyTorch
Lightning: SoundStream and EnCodec pipelines for waveform codecs, plus a
multi-speaker speech-activity codec (conv autoencoder + RVQ).

## Layout

| Path | Description |
| --- | --- |
| `soundstream_train.py` | SoundStream encoder / RVQ / decoder plus waveform and STFT discriminators, trained with a GAN objective (manual optimization, 24 kHz). |
| `encodec_train.py` | EnCodec model and multi-scale STFT discriminator with a balancer, trained with the EnCodec loss (24 kHz, 1 s segments). |
| `encodec_train_noadv.py` | Same EnCodec model trained without adversarial losses (time L1 + multi-scale mel + commitment only). |
| `soundstream_train_encodecloss.py` | SoundStream model trained with the EnCodec objective (time L1 + mel, MS-STFT hinge, feature matching, balancer). |
| `A_codec_train.py` | Multi-speaker speech-activity codec (conv AE + RVQ) over `(max_speakers, T)` activity grids from AMI/NOTSOFAR Lhotse cuts or a turn-taking simulator. |
| `A_codec_train_balanced.py` | Same model plus difficulty-balanced training: frame-level DER by difficulty/speaker count, difficulty-weighted sampling, MixAugment, boundary-weighted/focal losses, curriculum stages. |
| `docs/SoundStream-main/` | Reference SoundStream implementation (architecture and original training loop). |
| `docs/encodec-main/` | Reference EnCodec implementation from Meta. |
| `docs/snac-main/` | Reference SNAC implementation. |

## Requirements

```
torch
torchaudio
pytorch-lightning
vector-quantize-pytorch
numpy
lhotse
transformers
tqdm
```

## Data

The waveform codec scripts (`soundstream_*`, `encodec_*`) expect LibriTTS
(`train-clean-100` for training, `dev-clean` for validation) and read audio
with `torchaudio`.

The speech-activity codecs (`A_codec_train*.py`) read real AMI / NOTSOFAR
Lhotse cut manifests (mt-asr chunking) and/or a turn-taking conversation
simulator that generates the `(max_speakers, T)` activity grids.

Dataset paths are hard-coded for the Kaggle environment and should be
adjusted for local runs. Data and checkpoints are not included in this
repository.

## Usage

```bash
python soundstream_train.py
python encodec_train.py
python encodec_train_noadv.py
python soundstream_train_encodecloss.py
python A_codec_train.py
python A_codec_train_balanced.py
```

## Third-party code

Everything under `docs/` is reference code from upstream projects and keeps
its original license: EnCodec (MIT), SNAC (MIT), and the SoundStream
implementation it was derived from.
