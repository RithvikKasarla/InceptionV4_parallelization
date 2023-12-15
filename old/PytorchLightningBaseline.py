import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import pretrainedmodels
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize(299),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# Load CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

class InceptionNetV4Model(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(InceptionNetV4Model, self).__init__()
        self.model = pretrainedmodels.inceptionv4(pretrained=None)
        in_features = self.model.last_linear.in_features
        self.model.last_linear = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    def train_dataloader(self):
        return trainloader

    def val_dataloader(self):
        return testloader

if __name__ == '__main__':
    model = InceptionNetV4Model()
    trainer = pl.Trainer(max_epochs=10, gpus=2)  
    trainer.fit(model)
