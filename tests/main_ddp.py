import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet34
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import argparse
import wandb

def setup(rank, world_size, ip_file: str):
    print("hey")
    with open(ip_file, 'r') as file:
        master_addr = file.read().strip()

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '12355'  # Ensure this port is open and the same across all nodes
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank} initialized with master at {master_addr}")


def cleanup():
    dist.destroy_process_group()


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def train(rank, world_size, ip_file: str):

    rank = int(rank) - 1
    if rank == 0:
        os.environ["WANDB__SERVICE_WAIT"]="200"
        wandb.init(project="dist_play", entity="cisl-bu")
    world_size = int(world_size)
    setup(rank, world_size, ip_file=ip_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and optimizers
    model = resnet34().to(device)
    model = DDP(model, device_ids=None)

    dataset = CIFAR10(root='data', train=True, download=True,
                      transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))]))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=128)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10000):  # loop over the dataset multiple times
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for _, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if rank == 0:
            wandb.log({"total_loss": total_loss})
    cleanup()

if __name__ == "__main__":
    print("entering main")

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Available GPUs:", torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank')
    parser.add_argument('--world_size')
    args = parser.parse_args()
    print("Rank:", args.rank)

    print("starting training")
    train(args.rank, args.world_size, ip_file="networking/master_ip.txt")
