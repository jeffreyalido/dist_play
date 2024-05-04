#!/bin/bash -l

# Set SCC project
#$ -P tianlabdl

# Specify hard time limit for the job.
#$ -l h_rt=12:00:00

# Combine output and error files into a single file
#$ -j y

# Specify number of cores
#$ -pe omp 1

# Specify the array job range
#$ -t 1-2

# Specify the number of GPUs
#$ -l gpus=1

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load python3/3.10.12

# For networking stuff
IP_FILE="dist_play/master_ip.txt"

# Check if this is the first task
if [ $SGE_TASK_ID -eq 1 ]; then
  rm $IP_FILE
  python dist_play/head_ip.py $IP_FILE
  echo "IP address set by task $SGE_TASK_ID"
fi

# Wait to ensure the IP file is written by the first task
while [ ! -f $IP_FILE ]; do
  echo "Waiting for the master IP file..."
  sleep 3
done

echo "Master IP file found. Proceeding with training."
echo "Master IP: $(cat $IP_FILE)"
echo "Master Port: 12355"


source .venv/bin/activate
python tests/main.py --rank $SGE_TASK_ID --world_size 2
