# inviting the homies top the party  ;)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import random

# Download training dataset (PIL image objects)
image_dataset = MNIST(root='data/', download=True)

# Training dataset converted to tensors
tensor_training_dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

# data example
image, label = image_dataset[random.randint(0, 60001)]
print(f'Label: {label}')
plt.imshow(image, cmap='gray')
plt.show()

img_tensor, label = tensor_training_dataset[0]

# splitting existing dataset into Training and validation sets
train_ds, val_ds = random_split(tensor_training_dataset, [50000, 10000])

# setting batch size for training model efficiency
batch_size = 128

# load data
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

# create Logistic regression model
input_size = 28*28
output_size = 10

""" model = nn.Linear(input_size, output_size)

for images, labels in train_loader:
    print(labels)

    # reshape input matrix into a vector of size 784
    images.reshape(batch_size, input_size)
    data = model(images) """

# extend the nn.module class to create a Mnist class with all required functionality.

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, xb):
        xb = xb.reshape(-1, input_size)
        out = self.linear(xb)
        return out
    
model = MnistModel()

for images, labels in train_loader:
    print(images.shape)
    outputs = model(images)
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)

# Apply softmax for each output row
probs = F.softmax(outputs, dim=1)

# Look at sample probabilities
print("Sample probabilities:\n", probs[:2].data)

# Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]).item())

max_probs, preds = torch.max(probs, dim=1)
print(preds)
print(max_probs)
print(labels)

print(torch.sum(preds == labels))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

print(accuracyout)