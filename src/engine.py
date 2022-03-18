import os
from loguru import logger
import tqdm
from config import *
from dataset import *
import wandb
from torch import nn
import torch.optim as optim
from model import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = NetModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=learning_rate, weight_decay=weight_decay)

wandb.init(project="cifar10-pytorch")
wandb.watch(net, criterion, log="all")

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

columns=['image','label']
data = []

for i, batch in enumerate(trainloader, 0):
    inputs, labels = batch[0], batch[1]
    for j, image in enumerate(inputs,0):
        data.append([wandb.Image(image),classes[labels[j].item()]])
    break

table= wandb.Table(data=data, columns=columns)
wandb.log({"cifar10_images": table})

def train(model, device, trainloader, optimizer, criterion):
  for epoch in range(2):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

      inputs, labels = data[0].to(device), data[1].to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

      if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 2000))
        wandb.log({'epoch': epoch+1, 'loss': running_loss/2000})
        running_loss = 0.0

  print('Finished Training')