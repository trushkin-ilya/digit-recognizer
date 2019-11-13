import torch
import torch.utils.data
from torchvision import datasets, transforms
import PIL
import numpy as np
import pandas as pd

class MNISTTrainDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.transform = transforms.Compose(
                            [transforms.ToPILImage(), transforms.RandomAffine(20,(0.11,0.11)),
                             transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        df = pd.read_csv(path)
        self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
        self.y = torch.from_numpy(df.iloc[:,0].values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]

class MNISTTestDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        df = pd.read_csv(path)
        self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
            return self.transform(self.X[idx])
