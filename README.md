# Distributed PyTorch Training on BU's SGE Cluster

Simple distributed training scripts using PyTorch's `DistributedDataParallel` on clusters managed by SGE (qsub).

## Quick Start

### Multi-Node Training
```bash
qsub ddp_multi_node.sh
```

### Single-Node Multi-GPU Training  
```bash
qsub ddp_single_node.sh
```

## Scripts

- `ddp_main.py` - Multi-node distributed training across compute nodes
- `ddp_main_one_node.py` - Single-node multi-GPU training

## Notes

- Multi-node setup uses file-based coordination for master IP discovery
- Works with heterogeneous SGE GPU assignments
- Startup overhead is noticeable for multi-node jobs
- Single-node approach is faster for smaller scale training
- Ray is a wrapper around torch DDP. 

## Dependencies

- PyTorch with CUDA
- SGE cluster access with GPU nodes
