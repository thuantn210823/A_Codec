# A_Codec

Neural audio codec training code for the LastDiar project: a SoundStream
implementation and an EnCodec training pipeline, both built on
PyTorch Lightning.

## Layout

| Path | Description |
| --- | --- |
| `soundstream_train.py` | SoundStream encoder / RVQ / decoder plus waveform and STFT discriminators, trained with a GAN objective (manual optimization, 24 kHz). |
| `encodec_train.py` | EnCodec model and multi-scale STFT discriminator with a balancer, trained with the EnCodec loss (24 kHz, 1 s segments). |
| `docs/SoundStream-main/` | Reference SoundStream implementation (architecture and original training loop). |
| `docs/encodec-main/` | Reference EnCodec implementation from Meta. |
| `docs/snac-main/` | Reference SNAC implementation. |

## Requirements

```
torch
torchaudio
pytorch-lightning
vector-quantize-pytorch
tqdm
```

## Data

Both training scripts expect LibriTTS (`train-clean-100` for training,
`dev-clean` for validation) and read audio with `torchaudio`. The dataset
paths are hard-coded for the Kaggle environment and should be adjusted for
local runs. Data and checkpoints are not included in this repository.

## Usage

```bash
python soundstream_train.py
python encodec_train.py
```

## Third-party code

Everything under `docs/` is reference code from upstream projects and keeps
its original license: EnCodec (MIT), SNAC (MIT), and the SoundStream
implementation it was derived from.
