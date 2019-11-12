import torch
import torch.utils.data
from torchvision import datasets, transforms
import PIL
import pandas as pd



train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomRotation(45, resample=PIL.Image.BILINEAR), # аугментация: поворот до 45°
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

test_df = pd.read_csv("test.csv")
test_images = (test_df.iloc[:,:].values).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28)
test_images_tensor = torch.tensor(test_images)/255.0
kaggle_loader=torch.utils.data.DataLoader(test_images_tensor)
