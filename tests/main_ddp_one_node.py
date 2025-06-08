import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet34
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.multiprocessing as mp
import wandb


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    print(f"Rank {rank} initialized.")


def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    print(f"Starting rank {rank} on GPU {rank}")
    setup(rank, world_size)

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    if rank == 0:
        os.environ["WANDB__SERVICE_WAIT"] = "200"
        wandb.init(project="dist_play", entity="cisl-bu")

    model = resnet34().to(device)
    model = DDP(model, device_ids=[rank])

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

    duration = time.time() - start_time

    if rank == 0:
        with open("runtime_log_singlenode.txt", "a") as f:
            f.write(f"Total training time: {duration:.2f} seconds\n")

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"Launching training on {world_size} GPUs")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
