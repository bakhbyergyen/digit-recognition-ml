import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from utils import clean_up, create_tar_gz, save_model, upload_to_s3

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--bucket_name", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--version", type=str, required=True)
args = parser.parse_args()

mnist = load_dataset("mnist")

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset class
class MNISTDataset(Dataset):
    def __init__(self, split):
        self.data = mnist[split]
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        label = self.data[idx]["label"]
        image = self.transform(np.array(image))
        return image, label

train_dataset = MNISTDataset("train")

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

class AmptalkModel(nn.Module):
    def __init__(self):
        super(AmptalkModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model, Loss, Optimizer
model = AmptalkModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

n_epochs = 3


if __name__ == '__main__':
    
    # Version
    print(f"Training version {args.version}")
    # Training loop
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}")
                running_loss = 0.0

    perfix_path = '/opt/ml/model'
    os.makedirs(perfix_path, exist_ok=True)
    model_path = os.path.join(perfix_path, 'model.pth')
    tar_path = os.path.join(perfix_path, 'model.tar.gz')

    # Save model
    save_model(model, model_path)
    # Create tar.gz file
    create_tar_gz(model_path, tar_path)
    # Upload to S3 format "model_/model.tar.gz"
    s3_path = f"{args.output_path}_{args.version}/model.tar.gz"
    upload_to_s3(tar_path, args.bucket_name, s3_path)
    # Clean up local files
    clean_up(model_path, tar_path)
