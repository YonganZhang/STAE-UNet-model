
import torch
from torch.utils.data import Dataset
import numpy as np

class RockDataset(Dataset):
    def __init__(self, data_paths, labels):
        self.data_paths = data_paths
        self.labels = labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        volume = np.load(self.data_paths[idx])  # 假设存成 .npy 格式
        volume = torch.from_numpy(volume).float().unsqueeze(0)  # [1, D, H, W]
        label = torch.tensor(self.labels[idx]).float()
        return volume, label
