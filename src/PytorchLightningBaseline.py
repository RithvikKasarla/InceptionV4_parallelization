import time
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader
import timm
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from codecarbon import EmissionsTracker


class InceptionNetV4Model(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(InceptionNetV4Model, self).__init__()
        self.model = timm.create_model('inception_v4', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        print(f"Sub-batch size for GPU {self.global_rank}: {inputs.size(0)}") 
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    def train_dataloader(self):
        train_sampler = DistributedSampler(trainset, shuffle=True)
        return DataLoader(trainset, batch_size=32, num_workers=4, sampler=train_sampler)

    def val_dataloader(self):
        val_sampler = DistributedSampler(testset, shuffle=False)
        return DataLoader(testset, batch_size=32, num_workers=4, sampler=val_sampler)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    model = InceptionNetV4Model()

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu',
        devices=2,
        strategy='ddp'
    )
    
    print("Training Started")
    start_time = time.time()
    tracker = EmissionsTracker()
    tracker.start()

    trainer.fit(model, trainloader, testloader)
    end_time = time.time()
    emissions = tracker.stop()

    print("Total training time:", end_time - start_time,"seconds")
    print("Estimated CO2 emissions for training: ", emissions,"kg")


    trainer.save_checkpoint("inception_v4_trained.ckpt")
