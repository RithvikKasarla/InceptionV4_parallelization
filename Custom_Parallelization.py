import torch
import torchvision
import torchvision.transforms as transforms
import pretrainedmodels
import torch.nn.functional as F
from mpi4py import MPI

MPI.Init()
# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data Transforms
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load and Partition CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
subset_indices = torch.arange(len(trainset))
subset_size = len(subset_indices) // size
start_idx = rank * subset_size
end_idx = start_idx + subset_size if rank != size - 1 else len(subset_indices)
trainset_subset = torch.utils.data.Subset(trainset, subset_indices[start_idx:end_idx])
trainloader = torch.utils.data.DataLoader(trainset_subset, batch_size=32, shuffle=True, num_workers=2)

# InceptionNet-V4 Model
class InceptionNetV4Model(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionNetV4Model, self).__init__()
        self.model = pretrainedmodels.inceptionv4(pretrained=None)
        in_features = self.model.last_linear.in_features
        self.model.last_linear = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Gradient Synchronization Function
def all_reduce_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            comm.Allreduce(MPI.IN_PLACE, param.grad.data, op=MPI.SUM)

# Training Loop
model = InceptionNetV4Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Set the number of epochs

for epoch in range(num_epochs):
    total_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        all_reduce_gradients(model)
        optimizer.step()
        total_loss += loss.item()

        if i % 100 == 99:  # print every 100 mini-batches
            print(f"Rank {rank}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {total_loss / 100:.2f}")
            total_loss = 0.0

