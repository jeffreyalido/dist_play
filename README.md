# learning notes for torch's `DistributedDataParallel` 
just me playing around with `torch`'s distruted package for distributed training on an SGE-managed computer cluster. 

trying to get a `process_group` across heterogeneous machines with SGE array jobs where the job scheduler gives you random GPU devices on a node. but realized you don't need to set `device_id` when instantiating the DDP model.

the startup overhead takes quite a while tho. 
