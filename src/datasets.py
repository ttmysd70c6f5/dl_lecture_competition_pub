import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint


class ThingsMEGDataset(torch.utils.data.Dataset):
    # Train = ThingsMEGDataset("train",data_dir)
    # 
    # Methods:
    # Train.split: data type
    # Train.num_classes: number of classes
    # Train.X: data [n, ch, seq]
    # Train.subject_idxs: subject index for each sample
    # Train.y: true labels
    
    def __init__(self, split: str, data_dir: str = "data", trans = None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        self.transform = trans
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.transform:
            x = self.transform(eeg=x)['eeg']
            
        if hasattr(self, "y"):
            return x, self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]