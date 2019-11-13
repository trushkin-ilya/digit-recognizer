import torch
import torch.optim as optim
from lenet import LeNetEnsemble
from data_loader import MNISTTrainDataset, MNISTTestDataset
import torch.nn.functional as F
import os
from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch.utils.data

import wandb

import argparse

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, device, test_loader, use_wandb):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        predictions = make_predictions(model, test_loader)
        print(predictions)
    results = pd.Series(np.array(predictions, dtype = np.int32),name="Label")
    submission = pd.concat([pd.Series(range(1,len(results)+1), dtype=np.int32, name = "ImageId"),results],axis = 1)
    submission = submission.astype(np.int32)
    submission.to_csv("/content/gdrive/My Drive/12345/predictions.csv",index=False)
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if use_wandb:
        wandb.log({"Test Accuracy": 100. * correct / len(test_loader.dataset), "Test Loss": test_loss})

def make_predictions(model,data_loader):
    model.eval()
    test_preds = torch.LongTensor()
    
    for i, data in enumerate(data_loader):
        data = data.unsqueeze(1)
        
        if torch.cuda.is_available():
            data = data.cuda()
            
        output = model(data)
        
        preds = output.cpu().data.max(1, keepdim=True)[1][0]
        test_preds = torch.cat((test_preds, preds), dim=0)
        
    return test_preds

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Trains the 15 LeNet ensemble.")
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--momentum", type=float, default=0.5)
    argparser.add_argument("--use-wandb", type=bool, default=False)
    args = argparser.parse_args()
    lr = args.lr
    epochs = args.epochs
    momentum = args.momentum
    use_wandb= args.use_wandb
    if use_wandb:
        wandb.init(project="digit-recognizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(dataset=MNISTTrainDataset("train.csv"), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=MNISTTestDataset("test.csv"),batch_size=64, shuffle=False)
    model = LeNetEnsemble(15, device)
    if use_wandb:
        wandb.watch(model, log="all")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):    
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if use_wandb:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
