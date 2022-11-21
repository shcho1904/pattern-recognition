import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.activation import ReLU
import torch.nn.functional as F  # useful stateless functions
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
            
# 시드설정
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

dtype = torch.float32    
training_epochs = 15
batch_size = 100
hidden_layer_size = 1024
print_every = 100
learning_rate = 1e-2
train_loss_history = []
val_loss_history = []
train_accuracy = []
val_accuracy = []
skip = []

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(), 
                          #transform은 이미지를 tensor에 맞게 조정하기 위하여 생성
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

train_size = int(0.9*len(mnist_train))
val_size = len(mnist_train) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(mnist_train, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          drop_last=True )
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          drop_last=True )
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          drop_last=True )

def check_accuracy_part(loader, model, loss_history, acc_history):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        loss = F.cross_entropy(scores, y)
        
        loss_history.append(loss.item())
        acc_history.append(acc)
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_part(model, optimizer, epochs=15):
    model = model.to(device=device)
    for e in range(epochs):
        for t, (x,y) in enumerate(train_loader):
            model.train() #put model to training mode
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype = torch.long)
            
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
        check_accuracy_part(train_loader, model, train_loss_history, train_accuracy)
        check_accuracy_part(val_loader, model, val_loss_history, val_accuracy)
    
            

model = nn.Sequential(
    Flatten(),
    nn.Linear(28*28, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, 10),
    )

optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
train_part(model, optimizer)


check_accuracy_part(test_loader, model, skip, skip)