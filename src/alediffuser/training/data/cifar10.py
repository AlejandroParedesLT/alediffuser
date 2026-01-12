import torchvision
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Any, Tuple, List

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
        """Initialize additional attributes after dataclass init"""
        self.to_pil = ToPILImage()
    
    def transform_to_numpy(self, tensor):
        """Convert normalized tensor back to numpy array [0, 255]"""
        return (tensor * 127.5 + 127.5).long().clip(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    
    def transform_to_pil(self, tensor):
        """Convert normalized tensor back to PIL Image"""
        # Denormalize from [-1, 1] to [0, 1]
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        return self.to_pil(tensor.cpu())
    

def transformHelper(tensor):
    return (tensor*127.5+127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy()

def getCIFAR10(
    path_to_store_dataset:str='./datasets',
    batch_size:int=128,
    num_workers:int=2
):
    transform_to_tensor=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train_dataset=torchvision.datasets.CIFAR10(root=path_to_store_dataset,download=True,transform=transform_to_tensor)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    
    
    val_dataset=torchvision.datasets.CIFAR10(root=path_to_store_dataset,download=True,transform=transform_to_tensor,train=False)
    val_dataloader=DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    
    return DataPack(
        train_dataset=train_dataset,
        train_dataloader=train_dataloader,
        val_dataset=val_dataset,
        val_dataloader=val_dataloader,
        transform_to_tensor=transform_to_tensor
    )