import torch
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
import tqdm
from alediffuser.models.ddpm import DDPM
# from alediffuser.models.unet import UNet
from alediffuser.training.data.cifar10 import getCIFAR10


def train(
        *,
        model: DDPM,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.MSELoss,
        epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        path:str='checkpoint',
        device: str='cuda',
    ):
    training_losses=[]
    val_losses=[]
    for epoch in range(epochs):
        model.train()
        training_loss=0
        val_loss=0
        pbar=tqdm.tqdm(train_dataloader)
        for index,(image,label) in enumerate(pbar):
            optimizer.zero_grad()
            image=image.to(device)
            label=label.to(device)
            # Perform forward pass
            pred_noise,noise=model(image,label)

            loss=criterion(
                pred_noise,
                noise
            )
            loss.backward()
            optimizer.step()
            training_loss+=loss.item()
            pbar.set_description(f"Loss for epoch {epoch}: : {training_loss / (index + 1):.4f}")
        model.eval()
        with torch.no_grad():
            for (imgs,labels) in val_dataloader:
                imgs=imgs.to(device)
                labels=labels.to(device)
                pred_noise,noise=model(imgs,labels)
                val_loss+=loss.item()
            training_losses.append(training_loss/len(val_dataloader))
            val_losses.append(val_loss/len(val_dataloader))
            pbar.set_description(f"Validation Loss for epoch {epoch}: : {val_loss}")
        if epoch%25==0:
            pbar.set_description(f"Saving model {epoch}: : {val_loss}")
            torch.save(model.state_dict(),path)
        return training_losses,val_losses