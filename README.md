# fans_codex

This repository provides a small utility to visualize how diffusion noise
impacts different image frequencies over time. It loads a pretrained CIFAR-10
DDPM from Hugging Face, estimates the signal-to-noise ratio (SNR) for both the
forward and reverse processes, and saves heatmaps with contour lines at -10dB,
-5dB, 0dB, 5dB, and 10dB.

## Setup

Install Python 3.10+ and the required dependencies:

```bash
pip install diffusers torch torchvision matplotlib
```

> The script will download CIFAR-10 automatically (for Monte Carlo PSD
> estimation) and pull the `google/ddpm-cifar10-32` weights from Hugging Face.

## Usage

Generate the heatmaps with default parameters:

```bash
python snr_heatmap.py
```

Key options:

- `--num-samples`: number of CIFAR-10 images used to estimate the clean power
  spectrum (default: 256).
- `--num-timestep-bins`: how many evenly spaced timesteps to visualize across
  the diffusion schedule (default: 100).
- `--num-freq-bins`: number of radial frequency bins in the heatmap (default: 16).
- `--device`: set to `cuda` to leverage a GPU if available.

Outputs are saved to the `artifacts/` directory as `forward_snr.png` and
`reverse_snr.png`.
