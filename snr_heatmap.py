"""
Generate SNR heatmaps for forward and reverse diffusion processes on CIFAR-10.

The script loads a pretrained CIFAR-10 DDPM from Hugging Face, estimates the
frequency-domain signal-to-noise ratio (SNR) over timesteps, and plots heatmaps
with contour lines at standard dB levels.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import torch
from diffusers import DDPMPipeline, DDPMScheduler
from torchvision import datasets, transforms


@dataclass
class Config:
    model_id: str = "google/ddpm-cifar10-32"
    batch_size: int = 64
    num_samples: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of timesteps to visualize (evenly spaced across training steps)
    num_timestep_bins: int = 100
    # Frequency bins for radial averaging (<= image size // 2)
    num_freq_bins: int = 16
    contour_levels: Tuple[int, ...] = (-10, -5, 0, 5, 10)
    output_dir: Path = Path("artifacts")


# ---------- Utilities ----------

def get_dataset(cfg: Config) -> torch.Tensor:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    samples = []
    with torch.no_grad():
        for batch, _ in loader:
            samples.append(batch)
            if len(torch.cat(samples)) >= cfg.num_samples:
                break
    data = torch.cat(samples)[: cfg.num_samples]
    return data


def radial_psd(batch: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Compute radial power spectral density for a batch of images.

    Args:
        batch: Tensor [B, C, H, W] in [-1, 1].
        num_bins: Number of radial frequency bins.
    Returns:
        PSD tensor [num_bins].
    """
    b, c, h, w = batch.shape
    # Convert to grayscale to simplify averaging across channels
    gray = batch.mean(dim=1, keepdim=True)
    # Shift FFT so zero-frequency is at center
    fft = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1))
    power = (fft.real ** 2 + fft.imag ** 2).mean(dim=1)  # [B, H, W]

    yy, xx = torch.meshgrid(
        torch.arange(h, device=batch.device) - h // 2,
        torch.arange(w, device=batch.device) - w // 2,
        indexing="ij",
    )
    radial = torch.sqrt(xx ** 2 + yy ** 2)
    max_radius = radial.max()
    bin_edges = torch.linspace(0, max_radius, steps=num_bins + 1, device=batch.device)

    psd = torch.zeros(num_bins, device=batch.device)
    for i in range(num_bins):
        mask = (radial >= bin_edges[i]) & (radial < bin_edges[i + 1])
        count = mask.sum()
        if count > 0:
            # Average over spatial positions for each sample then across the batch
            masked_power = (power * mask).sum(dim=(1, 2)) / count
            psd[i] = masked_power.mean()
    return psd


def schedule_alphas(scheduler: DDPMScheduler, num_bins: int) -> Tuple[torch.Tensor, torch.Tensor]:
    steps = torch.linspace(0, scheduler.config.num_train_timesteps - 1, steps=num_bins)
    steps = steps.round().long()
    alpha_cumprod = scheduler.alphas_cumprod.to(steps.device)
    return steps, alpha_cumprod[steps]


# ---------- Forward process ----------

def compute_forward_snr(cfg: Config, clean_psd: torch.Tensor, scheduler: DDPMScheduler) -> torch.Tensor:
    steps, alpha_bar = schedule_alphas(scheduler, cfg.num_timestep_bins)
    # SNR = (alpha_bar * C_i) / (1 - alpha_bar)
    snr = []
    for a_bar in alpha_bar:
        snr.append((a_bar * clean_psd) / (1 - a_bar))
    snr = torch.stack(snr)  # [T, F]
    snr_db = 10 * torch.log10(snr + 1e-12)  # avoid log(0)
    return steps, snr_db


# ---------- Reverse process ----------

def compute_reverse_snr(
    cfg: Config, model: DDPMPipeline, scheduler: DDPMScheduler
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = cfg.device
    num_steps = cfg.num_timestep_bins
    timesteps = torch.linspace(0, scheduler.config.num_train_timesteps - 1, steps=num_steps)
    timesteps = timesteps.round().long().to(device)

    # Initialize random noise
    sample = torch.randn(cfg.batch_size, model.unet.config.in_channels, 32, 32, device=device)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)

    reverse_snr = []
    with torch.no_grad():
        for t in timesteps:
            t_int = t.item()
            # Move current sample to correct noise level
            idx = (scheduler.timesteps == t_int).nonzero(as_tuple=False)
            if len(idx) == 0:
                # Adjust scheduler to ensure this timestep exists
                scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)
                idx = (scheduler.timesteps == t_int).nonzero(as_tuple=False)
            timestep = scheduler.timesteps[idx[0].item()]

            model_output = model.unet(sample, timestep).sample
            alpha_bar = scheduler.alphas_cumprod[timestep].to(device)
            signal = (sample - torch.sqrt(1 - alpha_bar) * model_output) / torch.sqrt(alpha_bar)
            signal = torch.sqrt(alpha_bar) * signal  # scale back to x_t contribution
            noise = torch.sqrt(1 - alpha_bar) * model_output

            signal_psd = radial_psd(signal, cfg.num_freq_bins)
            noise_psd = radial_psd(noise, cfg.num_freq_bins) + 1e-12
            reverse_snr.append(signal_psd / noise_psd)

            # Single reverse step to update sample
            step_result = scheduler.step(model_output, timestep, sample)
            sample = step_result.prev_sample

    reverse_snr = torch.stack(reverse_snr)
    snr_db = 10 * torch.log10(reverse_snr + 1e-12)
    return timesteps.cpu(), snr_db.cpu()


# ---------- Plotting ----------

def plot_heatmap(
    snr_db: torch.Tensor,
    timesteps: torch.Tensor,
    cfg: Config,
    title: str,
    filename: Path,
):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    extent = [timesteps[0].item(), timesteps[-1].item(), 0, cfg.num_freq_bins]
    plt.imshow(
        snr_db.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    plt.colorbar(label="SNR (dB)")
    cs = plt.contour(
        torch.linspace(timesteps[0], timesteps[-1], snr_db.shape[0]),
        torch.linspace(0, cfg.num_freq_bins - 1, cfg.num_freq_bins),
        snr_db.T,
        levels=cfg.contour_levels,
        colors="white",
        linewidths=0.7,
    )
    plt.clabel(cs, fmt="%d dB", colors="white")
    plt.xlabel("Timestep")
    plt.ylabel("Frequency bin (radial)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---------- Main ----------

def main(cfg: Config):
    device = torch.device(cfg.device)
    model = DDPMPipeline.from_pretrained(cfg.model_id)
    model.to(device)
    scheduler = model.scheduler

    data = get_dataset(cfg).to(device)
    clean_psd = radial_psd(data, cfg.num_freq_bins)

    steps, forward_snr_db = compute_forward_snr(cfg, clean_psd, scheduler)
    plot_heatmap(forward_snr_db.cpu(), steps.cpu(), cfg, "Forward diffusion SNR", cfg.output_dir / "forward_snr.png")

    timesteps, reverse_snr_db = compute_reverse_snr(cfg, model, scheduler)
    plot_heatmap(reverse_snr_db, timesteps, cfg, "Reverse diffusion SNR", cfg.output_dir / "reverse_snr.png")
    print(f"Saved plots to {cfg.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SNR heatmaps for diffusion.")
    parser.add_argument("--num-samples", type=int, default=Config.num_samples, help="Number of CIFAR-10 samples for PSD estimation")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--num-timestep-bins", type=int, default=Config.num_timestep_bins)
    parser.add_argument("--num-freq-bins", type=int, default=Config.num_freq_bins)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    args = parser.parse_args()

    cfg = Config(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_timestep_bins=args.num_timestep_bins,
        num_freq_bins=args.num_freq_bins,
        device=args.device,
        output_dir=args.output_dir,
    )
    main(cfg)
