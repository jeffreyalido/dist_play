import torch
from torchvision.models import resnet34
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import argparse

from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

import ray.train.torch

def train_func():

    # Model and optimizers
    model = resnet34()
    model = ray.train.torch.prepare_model(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    dataset = CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))]),
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    train_loader = ray.train.torch.prepare_data_loader(dataloader)

    # Training
    for epoch in range(10):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        for images, labels in train_loader:
            # This is done by `prepare_data_loader`!
            # images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")


def read_ip_address(file_path):
    try:
        with open(file_path, "r") as file:
            ip_address = file.read().strip()
        return ip_address
    except Exception as e:
        print(f"Failed to read the IP address from {file_path}: {e}")
        return None


if __name__ == "__main__":
    ip_file_path = "dist_play/master_ip.txt"
    ray_head_node_address = read_ip_address(ip_file_path)

    ray.init(address=f"{ray_head_node_address}:6379")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers")
    args = parser.parse_args()
    print(f"Number of workers: {args.num_workers}")

    scaling_config = ScalingConfig(num_workers=int(args.num_workers), use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

    ray.shutdown()
