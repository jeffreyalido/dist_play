#!/bin/bash
# Initialize a new sweep, and extract the agent command

# Define a unique job name
timestamp=$(date +"%Y%m%d%H%M%S%N")  # %N for nanoseconds
random_suffix=$(($RANDOM % 1000))  # Generates a random number between 0 and 999
jobname="play_DP_sup_peeps_${timestamp}_${random_suffix}"
# Create a new .qsub file with dynamic job details and torchrun command
qsub_file="/projectnb/tianlabdl/jalido/dist_play/qsubs/DP.qsub"
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

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : \$(date)"
echo "Job name : \$JOB_NAME"
echo "Job ID : \$JOB_ID  \$SGE_TASK_ID"
echo "=========================================================="

# Specify number of cores 16: a whole node with at least 128gb ram
#$ -pe omp 2
#$ -l mem_per_core=16G
#$ -t 1

#Specify the number of GPUs (1 is recommended!)
#$ -l gpus=2


module load python3/3.10.12
source .venv/bin/activate
export TRITON_CACHE_DIR=/scratch
python tests/main_DataParallel.py
EOF

# Submit the job
qsub -N "${jobname}" -o "$log_file" "$qsub_file"