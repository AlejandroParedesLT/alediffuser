from alediffuser.models.ddpm import DDPM
from alediffuser.models.unet import UNet
from alediffuser.training.data.cifar10 import getCIFAR10

from alediffuser.training.train import train
from torchvision.utils import save_image,make_grid
import matplotlib.pyplot as plt
from torch.optim import AdamW
import torch
import torch.nn as nn

import argparse
import json
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Training configuration - flexible to accept any config structure"""
    # Model architecture
    TIME: int = 1000
    PATH_TO_SAVE_MODEL: str = 'exp1_T1000_epochs100.pth'
    EPOCHS: int = 100
    device: str = 'cuda'
    batch_size: int = 32

    in_channels: int = 3
    out_channels: int = 3
    num_classes: int = 10
    recommended_steps: list = field(default_factory=lambda: [1, 2, 2, 2])
    recommended_attn_step_indexes: list = field(default_factory=lambda: [1, 2, 3, 4])
    p_cond: float = 0.2

    # Training hyperparameters
    lr: float = 0.001
    lr_warmup: float = 0.001
    alpha_max:float = 6e-4    # Peak learning rate
    alpha_min:float = 6e-6    # Minimum learning rate (alpha_max / 10)
    T_w:int = 1000          # Warmup iterations (10% of total)
    T_c:int = 15000         # Total iterations
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.001
    
    # Data parameters
    train_data: str = "./data/train.txt"
    val_data: Optional[str] = None

    
    # Output and logging
    ckpt_path: str = "./outputs"
    log_interval: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    batch_size: int = 1028
    max_iters: int = 10000
    prefix_name_experiment: str = 'experiment'
    
    # Hardware
    device: str = "cuda"
    mixed_precision: bool = False
    dtype: str = "float64"
    optimized: bool=False
    
    # Other
    seed: int = 42
    debug: bool = False


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Create config object with loaded values
    return TrainingConfig(**config_dict)


def override_config(config: TrainingConfig, overrides: dict) -> TrainingConfig:
    """Override config values with CLI arguments"""
    config_dict = asdict(config)
    
    # Only override non-None values
    for key, value in overrides.items():
        if value is not None and key in config_dict:
            config_dict[key] = value
            print(f"  Overriding {key}: {config_dict[key]}")
    
    return TrainingConfig(**config_dict)


def parse_args():
    """Parse command-line arguments for config overrides"""
    parser = argparse.ArgumentParser(
        description='LLM Training with Config File Management',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file (required in production)
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (YAML or JSON) - REQUIRED')
    
    # Common overrides for experiments
    parser.add_argument('--experiment_name', type=str,
                        help='Override experiment name')
    parser.add_argument('--batch_size', type=int,
                        help='Override batch size')
    parser.add_argument('--learning_rate', type=float,
                        help='Override learning rate')
    parser.add_argument('--num_epochs', type=int,
                        help='Override number of epochs')
    parser.add_argument('--output_dir', type=str,
                        help='Override output directory')
    
    # Quick toggles
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision training')
    
    # Resume training
    parser.add_argument('--resume_from', type=str,
                        help='Resume from checkpoint')
    
    # Device override
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'],
                        help='Override device')
    
    return parser.parse_args()



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

    data=getCIFAR10(batch_size=32)
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
    optimizer=AdamW(params=ddpm.parameters(),lr=config.lr,weight_decay=config.weight_decay,betas=(config.beta1,config.beta2),eps=config.eps)
    criterion=nn.MSELoss()

    _,val_losses=train(
        model=ddpm,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.EPOCHS,
        device=config.device,
        train_dataloader=data.train_dataloader,
        val_dataloader=data.val_dataloader,
        path=config.PATH_TO_SAVE_MODEL
    )
    path=config.PATH_TO_SAVE_MODEL
    torch.save(ddpm.state_dict(),path)
    plt.plot(val_losses,label='Validation loss')
    plt.legend()

    n_samples = 10
    x_t = ddpm.sample(
        n_samples=n_samples,
        size=data.train_dataset[0][0].shape,
        classes=[0,1,2,3,4,5,6,7,8,9],
        w=0.2
    )
    result = []
    for i in range(x_t.shape[0]):
        result.append(data.transform_to_pil(x_t[i]))

    grid = make_grid(x_t, nrow=10)
    save_image(grid, f"sample.png")

    cols = 5
    rows = (n_samples // cols) + (0 if n_samples % cols == 0 else 1)
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for i in range(len(result)):
        row = i // cols
        axs[row, i % cols].imshow(result[i], cmap='gray')


if __name__=='__main__':
    main()