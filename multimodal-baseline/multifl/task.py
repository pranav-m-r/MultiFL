"""multifl: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from multifl.dataloader import HatefulMemesDataset


fds = None  # Cache FederatedDataset


# Image transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="neuralcatcher/hateful_memes",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # Create dataset and loaders
    multimodal_dataset = HatefulMemesDataset(partition, tokenizer, transform=image_transform)
    # Divide data on each node: 80% train, 20% test
    train_size = int(0.8 * len(multimodal_dataset))
    test_size = len(multimodal_dataset) - train_size
    generator = torch.Generator().manual_seed(partition_id)
    train_dataset, test_dataset = torch.utils.data.random_split(multimodal_dataset, [train_size, test_size], generator=generator)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)
    return trainloader, testloader


def train_model(model, trainloader, epochs, device):
    """Train the model on the training set."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    total_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            texts = batch["text"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model({"image": images, "text": texts})
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return total_loss / len(trainloader)


def test_model(model, valloader, device):
    """Evaluate the model on the validation set."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in valloader:
            images = batch["image"].to(device)
            texts = batch["text"].to(device)
            labels = batch["label"].to(device)

            outputs = model({"image": images, "text": texts})
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return total_loss / len(valloader), accuracy


def get_weights(model):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
