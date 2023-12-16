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
from codecarbon import EmissionsTracker

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("I am Rank:", rank)
print("Size of MPI Tasks: ", size)

device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=size, rank=rank)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=2, sampler=train_sampler)

class InceptionNetV4Model(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionNetV4Model, self).__init__()
        self.model = timm.create_model('inception_v4', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

model = InceptionNetV4Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()

simulated_gpus = 2
sub_batch_size = 32 // simulated_gpus
tracker = EmissionsTracker()
tracker.start()

for epoch in range(1):
    train_sampler.set_epoch(epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # Splitting the batch into smaller sub-batches
        sub_batches = torch.chunk(inputs, simulated_gpus)
        sub_labels = torch.chunk(labels, simulated_gpus)

        for sim_gpu in range(simulated_gpus):
            if sim_gpu % size == rank:
                print("rank:", rank, "processing simulated GPU:", sim_gpu)

                sub_inputs = sub_batches[sim_gpu].to(device)
                sub_labels_j = sub_labels[sim_gpu].to(device)

                optimizer.zero_grad()

                outputs = model(sub_inputs)
                loss = criterion(outputs, sub_labels_j)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Synchronize after each simulated GPU
            comm.Barrier()

        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / (100 * simulated_gpus):.3f}")
            running_loss = 0.0

end_time = time.time()
emissions = tracker.stop()

print("Total training time:", end_time - start_time, "seconds")
print("Estimated CO2 emissions for training: ", emissions,"kg")

print('Finished Training')
