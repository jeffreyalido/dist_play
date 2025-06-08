# just playing around with distributed stuff for torch on BU's cluster

##  notes for torch's `DistributedDataParallel` 

trying to get a `process_group` across heterogeneous machines with SGE array jobs where the job scheduler gives you random GPU devices on a node. but realized you don't need to set `device_id` when instantiating the DDP model.

the startup overhead takes quite a while tho. 

one tricky thing was getting the master IP address for a compute node that we won't know until the job gets off the queue. workaround by writing the master ip address to a text file from which all the worker nodes can read

## notes for ray train with ray cluster

having trouble consistently connecting to the cluster spawned by the head node. probably something with my bash script.

there's some weird thing with reduplication. something is being repeated a couple thousand times across the cluster, need to investigate further.

the output logs are also messier.

not that much more abstracted than torch's `DDP` when using it on SGE-managed clusters. currently prefer `DDP`, but want to figure out why the startup takes a while.
