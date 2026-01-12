import argparse
import json
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List

from torchvision.utils import save_image,make_grid
import matplotlib.pyplot as plt
from torch.optim import AdamW
import torch
import torch.nn as nn


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

    dataset:str = 'CIFAR10'
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
    log_interval: int = 10
    save_steps: int = 10
    eval_steps: int = 10
    batch_size: int = 10
    max_iters: int = 10
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

import typing
import torch.optim as optim
def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    device: str | torch.device = None,
    strict: bool = True
):
    """
    Load model checkpoint and restore training state.
    
    Args:
        src: Path to checkpoint file or file-like object
        model: PyTorch model to load state into
        optimizer: Optional PyTorch optimizer to load state into
        device: Device to load checkpoint on (e.g., 'cuda', 'cpu')
        strict: Whether to strictly enforce state_dict keys match
    
    Returns:
        Tuple of (model, optimizer, iteration) if optimizer provided
        Tuple of (model, iteration) if optimizer is None
    """
    # Determine device
    if device is None:
        device = next(model.parameters()).device
    
    # Load checkpoint
    if isinstance(src, (str, os.PathLike)):
        # Load from file path
        chkpt = torch.load(src, map_location=device)
    elif hasattr(src, 'read'):
        # Load from file-like object
        chkpt = torch.load(src, map_location=device)
    else:
        raise TypeError(f"Unsupported input type: {type(src)}")
    
    # Load model state
    model.load_state_dict(chkpt, strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in chkpt:
        optimizer.load_state_dict(chkpt['optimizer'])
    
    # Get iteration
    iteration = chkpt.get('iteration', 0)
    
    print(f"Checkpoint loaded from iteration {iteration}")
    
    if optimizer is not None:
        return model, optimizer, iteration
    else:
        return model, iteration