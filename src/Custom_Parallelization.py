import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mpi4py import MPI
import os
import time
import timm

# Initialize MPI
# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set device based on MPI rank
device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

# Data Transforms
transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Load and Partition CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=size, rank=rank)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=2, sampler=train_sampler)

class InceptionNetV4Model(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionNetV4Model, self).__init__()
        # Load Inception-v4 model from timm without pretrained weights
        self.model = timm.create_model('inception_v4', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

model = InceptionNetV4Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training Loop

start_time = time.time()

for epoch in range(10):  # loop over the dataset multiple times
    train_sampler.set_epoch(epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

end_time = time.time()
print("Total training time:", end_time - start_time,"seconds")

print('Finished Training')

# MPI.Finalize()
