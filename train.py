import torch
import torch.optim as optim
from lenet import LeNetEnsemble
from data_loader import train_loader, test_loader
import torch.nn.functional as F
import os
from torch.autograd import Variable
import numpy as np
import pandas as pd

import wandb
wandb.init(project="digit-recognizer")

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
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    outputs=[]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            outputs.append(torch.argmax(output).cpu().item())
            correct += pred.eq(target.view_as(pred)).sum().item()
    results = pd.Series(np.array(outputs, dtype = np.int32),name="Label")
    submission = pd.concat([pd.Series(range(1,len(outputs)+1), dtype=np.int32, name = "ImageId"),results],axis = 1)
    submission = submission.astype(np.int32)
    submission.to_csv("/content/gdrive/My Drive/12345/predictions.csv",index=False)
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    wandb.log({"Test Accuracy": 100. * correct / len(test_loader.dataset), "Test Loss": test_loss})

def prediciton(data_loader):
    model.eval()
    test_pred = torch.LongTensor()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = Variable(data, volatile=True)
            if torch.cuda.is_available():
                data = data.cuda()

            output = model(data)

            pred = output.cpu().data.max(1, keepdim=True)[1]
            test_pred = torch.cat((test_pred, pred), dim=0)

    return test_pred

lr=0.01
epochs=1
momentum=0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeNetEnsemble(15,device)
wandb.watch(model, log="all")
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for epoch in range(1, epochs + 1):    
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
