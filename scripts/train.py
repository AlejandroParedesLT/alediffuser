from alediffuser.models.ddpm import DDPM
from alediffuser.models.unet import UNet
from alediffuser.training.data.cifar10 import getCIFAR10

from alediffuser.training.train import train
from torchvision.utils import save_image,make_grid
import matplotlib.pyplot as plt
from torch.optim import AdamW
import torch
import torch.nn as nn

def main():
    T=1000
    # dataset='mnist'
    PATH_TO_READY_MODEL=None
    PATH_TO_SAVE_MODEL='exp1_T1000_epochs100.pth'
    EPOCHS = 100
    device='cuda'
    data=getCIFAR10(batch_size=32)

    in_channels=3
    out_channels=3
    num_classes=10
    recommended_steps=(1,2,2,2)
    recommended_attn_step_indexes=[1,2,3,4]

    ddpm=DDPM(
        T=T,
        p_cond=0.2,
        eps_model=UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            T=T+1,
            num_classes=num_classes,
            steps=recommended_steps,
            attention_step_indexes=recommended_attn_step_indexes
        ),
        device=device
    )

    _,val_losses=train(
        model=ddpm,
        optimizer=AdamW(params=ddpm.parameters(),lr=2e-4),
        criterion=nn.MSELoss(),
        epochs=EPOCHS,
        device=device,
        train_dataloader=data.train_dataloader,
        val_dataloader=data.val_dataloader,
        path=PATH_TO_SAVE_MODEL
    )
    path=PATH_TO_SAVE_MODEL
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