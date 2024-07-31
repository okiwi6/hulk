import torch
import torch.nn.functional as F
from ssd.model.mnist_classifier import MnistClassifier
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

trainset = MNIST(root="data", download=True, train=True)
testset = MNIST(root="data", download=True, train=False)


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack([pil_to_tensor(img).float() / 255.0 for img in images])
    labels = torch.tensor(labels)
    return images, labels


trainloader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=64, shuffle=False, collate_fn=collate_fn)


model = MnistClassifier(10)
optimizer = Adam(model.parameters(), lr=0.007)

for images, labels in (pbar := tqdm(trainloader)):
    optimizer.zero_grad()

    predictions = model(images)
    loss = F.cross_entropy(predictions, labels)

    loss.backward()
    optimizer.step()

    accuracy = (predictions.argmax(dim=1) == labels).float().mean()
    pbar.set_description(f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
