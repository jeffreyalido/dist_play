import os
import torch

from torchvision.models import resnet34
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import argparse
import wandb


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def train():

    wandb.init(project="dist_play", entity="cisl-bu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and optimizers
    model = resnet34()
    model = torch.nn.DataParallel(model)

    dataset = CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))]),
    )
    dataloader = DataLoader(dataset, shuffle=True, batch_size=128)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10000):  # loop over the dataset multiple times
        total_loss = 0.0
        for _, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        wandb.log({"total_loss": total_loss})


if __name__ == "__main__":

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Available GPUs:", torch.cuda.device_count())
    parser = argparse.ArgumentParser()

    print("starting training")
    train()
