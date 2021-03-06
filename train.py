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

def predict(model, device, test_loader, use_wandb, output_dir):
    model.eval()
    preds = torch.LongTensor().to(device)
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            out_preds = output.argmax(dim=1, keepdim=True)
            preds = torch.cat((preds, out_preds.flatten()))
    results = pd.Series(np.array(preds.cpu(), dtype=np.int32), name="Label")
    submission = pd.concat([pd.Series(
        range(1, len(results)+1), dtype=np.int32, name="ImageId"), results], axis=1)
    submission = submission.astype(np.int32)
    submission.to_csv(output_dir + "/predictions.csv", index=False)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Trains the 15 LeNet ensemble.")
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--momentum", type=float, default=0.5)
    argparser.add_argument("--use-wandb", type=bool, default=False)
    argparser.add_argument("--output-dir", type=str, default=".")
    argparser.add_argument("--predict-every", type=int, default=1)
    args = argparser.parse_args()
    lr = args.lr
    epochs = args.epochs
    momentum = args.momentum
    use_wandb = args.use_wandb
    output_dir = args.output_dir
    predict_every = args.predict_every
    
    if use_wandb:
        wandb.init(project="digit-recognizer")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = torch.utils.data.DataLoader(
        dataset=MNISTTrainDataset("train.csv"), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=MNISTTestDataset("test.csv"), batch_size=64, shuffle=False)
    
    model = LeNetEnsemble(15, device)
    
    if use_wandb:
        wandb.watch(model, log="all")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        if epoch % predict_every == 1:
            predict(model, device, test_loader, use_wandb, output_dir)

    if use_wandb:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
