import torch
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
import tqdm
from alediffuser.models.ddpm import DDPM
# from alediffuser.models.unet import UNet
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR

def train(
        *,
        model: DDPM,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.MSELoss,
        scheduler: SequentialLR,
        epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        path:str='checkpoint',
        device: str='cuda',
        log_dir: str = 'runs',
    ):
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard writer
    training_losses=[]
    val_losses=[]
    best_val_loss = float('inf')  # Initialize best validation loss

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
            scheduler.step()
            training_loss+=loss.item()
            pbar.set_description(f"Loss for epoch {epoch}: : {training_loss / (index + 1):.4f}")
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                pred_noise, noise = model(imgs, labels)
                val_loss += criterion(pred_noise, noise).item()  # <-- compute actual loss here

        avg_train_loss = training_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        training_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        # Log losses to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        pbar.set_description(f"Validation Loss epoch {epoch}: {avg_val_loss:.4f}")

        # Save only if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), path)
            pbar.set_description(f"Validation improved, model saved at epoch {epoch}: {avg_val_loss:.4f}")
    return training_losses,val_losses