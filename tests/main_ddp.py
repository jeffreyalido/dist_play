import os
import time
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
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank} initialized with master at {master_addr}")
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, ip_file: str):
    rank = int(rank) - 1
    world_size = int(world_size)

    if rank == 0:
        os.environ["WANDB__SERVICE_WAIT"] = "200"
        wandb.init(project="dist_play", entity="cisl-bu")

    setup(rank, world_size, ip_file=ip_file)

    assert torch.cuda.device_count() == 1
    local_rank = 0
    device = torch.device("cuda", local_rank)

    model = resnet34().to(device)
    model = DDP(model, device_ids=[local_rank])

    dataset = CIFAR10(root='data', train=True, download=True,
                      transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))]))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=512)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    start_time = time.time()

    for epoch in range(100):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for _, data in enumerate(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            wandb.log({"total_loss": total_loss})

    end_time = time.time()
    duration = end_time - start_time

    if rank == 0:
        with open("runtime_log_multinode.txt", "a") as f:
            f.write(f"Total training time: {duration:.2f} seconds\n")

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
