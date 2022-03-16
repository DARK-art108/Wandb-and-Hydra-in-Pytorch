import torch
import torchvision
from engine import *
from model import *
from config import *
import torch.optim as optim
from dataset import trainloader, testloader, classes
import os

net = NetModel()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=learning_rate, weight_decay=weight_decay)

train(trainloader, criterion, net, device, optimizer)