import torchvision
import wandb
from engine import *
from model import *
from config import *
import torch.optim as optim
from dataset import trainloader, testloader, classes
import os

train(net, device, trainloader, optimizer, criterion)
