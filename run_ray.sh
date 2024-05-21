#!/bin/bash
# Initialize a new sweep, and extract the agent command

# Define a unique job name
timestamp=$(date +"%Y%m%d%H%M%S%N")  # %N for nanoseconds
random_suffix=$(($RANDOM % 1000))  # Generates a random number between 0 and 999
jobname="play_ray_sup_peeps_${timestamp}_${random_suffix}"
# Create a new .qsub file with dynamic job details and torchrun command
qsub_file="/projectnb/tianlabdl/jalido/dist_play/qsubs/ray.qsub"
log_file="/projectnb/tianlabdl/jalido/dist_play/logs/${jobname}.qlog"
# Create a new .qsub file with dynamic agent command and job details
cat <<EOF > "$qsub_file"
#!/bin/bash -l

# Set SCC project
#$ -P tianlabdl

# Specify hard time limit for the job.
#$ -l h_rt=3:00:00

# Combine output and error files into a single file
#$ -j y

# Specify number of cores
#$ -pe omp 2

# Specify the array job range
#$ -t 1-2

# Specify the number of GPUs
#$ -l gpus=1

echo "=========================================================="
echo "Start date : \$(date)"
echo "Job name : \$JOB_NAME"
echo "Job ID : \$JOB_ID  \$SGE_TASK_ID"
echo "=========================================================="

module load python3/3.10.12

# For networking stuff
IP_FILE="dist_play/master_ip.txt"

# Check if this is the first task
if [ \$SGE_TASK_ID -eq 1 ]; then
  rm \$IP_FILE
  python dist_play/head_ip.py \$IP_FILE
  echo "IP address set by task \$SGE_TASK_ID"
fi

# Wait to ensure the IP file is written by the first task
while [ ! -f \$IP_FILE ]; do
  echo "Waiting for the master IP file..."
  sleep 3
done

echo "Master IP file found. Proceeding with training."
echo "Master IP: \$(cat \$IP_FILE)"
echo "Master Port: 12355"

source .venv/bin/activate

# ray stuff
ray stop
ray disable-usage-stats
RAY_DEDUP_LOGS=0 # https://discuss.ray.io/t/how-do-i-disable-repeated-3x-across-cluster/11072/5

START_TIME=\$(date +%s)

# Initialize Ray - replace 'node1' with the actual hostname of your head node if needed
if [ \$SGE_TASK_ID -eq 1 ]; then
    ray start --head --port=6379 --num-gpus=1
    echo "Ray head node started on \$(hostname)"
else
    MASTER_IP=\$(cat \$IP_FILE)
    ray start --address=\$MASTER_IP:6379 --num-gpus=1
    echo "Connected to Ray head node at \$MASTER_IP"
fi

# Execute the Python script adapted for Ray
python tests/main_ray.py --num_workers 2

END_TIME=\$(date +%s)

ray stop

# Calculate the duration
DURATION=\$((END_TIME - START_TIME))

echo "=========================================================="
echo "End date : \$(date)"
echo "Execution time: \$DURATION seconds"
echo "=========================================================="

EOF

# Submit the job
qsub -N "${jobname}" -o "$log_file" "$qsub_file"