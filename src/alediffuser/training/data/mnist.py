import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from typing import Any
from torchvision.transforms import ToPILImage
import torch

@dataclass
class DataPack:
    train_dataset: Dataset
    train_dataloader: DataLoader
    val_dataset: Dataset
    val_dataloader: DataLoader
    transform_to_tensor: Any

    def __post_init__(self):
        self.to_pil = ToPILImage()

    def transform_to_numpy(self, tensor):
        # works for 1-channel MNIST; squeezes channel dimension if present
        arr = (tensor * 127.5 + 127.5).long().clip(0, 255).detach().cpu()
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        else:
            arr = arr.permute(1, 2, 0)
        return arr.numpy()

    def transform_to_pil(self, tensor):
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        return self.to_pil(tensor.cpu())


def getMNIST(
    path_to_store_dataset: str = "./datasets",
    batch_size: int = 128,
    num_workers: int = 2
):
    transform_to_tensor = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # single grayscale channel
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=path_to_store_dataset,
        train=True,
        download=True,
        transform=transform_to_tensor
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_dataset = torchvision.datasets.MNIST(
        root=path_to_store_dataset,
        train=False,
        download=True,
        transform=transform_to_tensor
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return DataPack(
        train_dataset=train_dataset,
        train_dataloader=train_dataloader,
        val_dataset=val_dataset,
        val_dataloader=val_dataloader,
        transform_to_tensor=transform_to_tensor
    )

def getFashionMNIST(
    path_to_store_dataset: str = "./datasets",
    batch_size: int = 128,
    num_workers: int = 2
):
    transform_to_tensor = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # single grayscale channel mapped to [-1, 1]
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root=path_to_store_dataset,
        train=True,
        download=True,
        transform=transform_to_tensor
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_dataset = torchvision.datasets.FashionMNIST(
        root=path_to_store_dataset,
        train=False,
        download=True,
        transform=transform_to_tensor
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return DataPack(
        train_dataset=train_dataset,
        train_dataloader=train_dataloader,
        val_dataset=val_dataset,
        val_dataloader=val_dataloader,
        transform_to_tensor=transform_to_tensor
    )
