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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


from alediffuser.models.ddpm import DDPM
from alediffuser.models.unet import UNet
from alediffuser.training.data.cifar10 import getCIFAR10
from alediffuser.training.data.mnist import getMNIST,getFashionMNIST
from alediffuser.training.train import train
from alediffuser.utils import load_config,override_config,parse_args
from alediffuser.memory_utils import get_single_sample_memory

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

    if config.dataset=='CIFAR10':
        data=getCIFAR10(batch_size=config.batch_size)
    elif config.dataset=='MNIST':
        data=getMNIST(batch_size=config.batch_size)
    elif config.dataset=='FASHIONMNIST':
        data=getFashionMNIST(batch_size=config.batch_size)
    else:
        data=getCIFAR10(batch_size=config.batch_size)

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

    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=config.lr_warmup,
        end_factor=1.0,
        total_iters=config.T_w 
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS - config.T_w,
        eta_min=config.alpha_min # Minimum LR
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.T_w]
    )

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_params(ddpm)
    print(f"Total parameters: {total_params:,}")
    
    print(get_single_sample_memory(model=ddpm,image_shape=(1,32,32),dtype=dtype_args))

    _,val_losses=train(
        model=ddpm,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
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
    plt.savefig()

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