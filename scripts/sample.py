import argparse
import json
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import save_image,make_grid

from alediffuser.models.ddpm import DDPM
from alediffuser.models.unet import UNet
from alediffuser.utils import load_config,override_config,parse_args,load_checkpoint
from alediffuser.training.data.mnist import getMNIST


def main():
    """Main function - config-file first approach"""
    args = parse_args()
    
    # Load base configuration from file
    config = load_config(args.config)
    
    # Apply CLI overrides if provided
    overrides = {k: v for k, v in vars(args).items() if k != 'config'}
    if any(v is not None for v in overrides.values()):
        print("\nApplying CLI overrides:")
        config = override_config(config, overrides)
    
    # Validate configuration
    # validate_config(config)
    
    dtype_args=None
    if config.dtype=='float64':
        dtype_args=torch.float64
    elif config.dtype=='float32':
        dtype_args=torch.float32
    else:
        dtype_args=torch.float16

    data=getMNIST(batch_size=32)
    
    ddpm=DDPM(
        T=config.TIME,
        p_cond=config.p_cond,
        eps_model=UNet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            T=config.TIME+1,
            num_classes=config.num_classes,
            steps=config.recommended_steps,
            attention_step_indexes=config.recommended_attn_step_indexes,
            device=config.device,
            dtype=dtype_args
        ),
        device=config.device,
        dtype=dtype_args
    )
    current_dir = Path(__file__).parent
    assets_dir = current_dir.parent / "assets"

    ddpm,_=load_checkpoint(assets_dir / config.PATH_TO_SAVE_MODEL ,ddpm)
    n_samples = 10
    x_t = ddpm.sample(
        n_samples=n_samples,
        size=data.train_dataset[0][0].shape,
        # classes=[0,1,2,3,4,5,6,7,8,9],
        # w=0.2
    )
    result = []
    for i in range(x_t.shape[0]):
        result.append(data.transform_to_pil(x_t[i]))

    x_t_denorm = x_t * 0.5 + 0.5
    x_t_denorm = torch.clamp(x_t_denorm, 0, 1)

    # Create a simple grid plot with matplotlib
    n_samples = x_t.shape[0]
    cols = 10
    rows = (n_samples + cols - 1) // cols  # Ceiling division

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))

    # Flatten axs if it's a 2D array
    if rows == 1:
        axs = axs.reshape(1, -1)
    elif cols == 1:
        axs = axs.reshape(-1, 1)

    for i in range(n_samples):
        row = i // cols
        col = i % cols
        
        # Get the image and move to CPU
        img = x_t_denorm[i].cpu()
        
        # If grayscale (1 channel), squeeze it
        if img.shape[0] == 1:
            img = img.squeeze(0)
        else:
            # If RGB (3 channels), permute to (H, W, C)
            img = img.permute(1, 2, 0)
        
        # Plot
        axs[row, col].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axs[row, col].axis('off')

    # Turn off any extra subplots
    for i in range(n_samples, rows * cols):
        row = i // cols
        col = i % cols
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(assets_dir / f"samples_grid_{config.PATH_TO_SAVE_MODEL}.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    main()