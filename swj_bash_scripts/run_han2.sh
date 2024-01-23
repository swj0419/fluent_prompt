#!/bin/bash
#SBATCH --job-name=hrpt
#SBATCH --partition=learnaccel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=volta32gb
#SBATCH --time=2-0:00:00
#SBATCH --chdir=/private/home/swj0419/i-am-a-dog/openprompt-clone/swj_bash_scripts
#SBATCH --output=/private/home/swj0419/i-am-a-dog/openprompt-clone/sbatch_run_script/report/dog_7-31/%j.out
#SBATCH -a 0-3:1

# Activate the environment
# learnfair learnlab
source ~/.bashrc
conda activate ptbase

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode

    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array

    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

# Check if the output folder exists at all. We could remove the folder in that case.
if [ -d  "$JOB_OUTPUT" ]
then
    echo "Folder exists, but was unfinished or is ongoing (no results.json)."
    echo "Starting job as usual"
    # It might be worth removing the folder at this point:
    # echo "Removing current output before continuing"
    # rm -r "$JOB_OUTPUT"
    # Since this is a destructive action it is not on by default
fi


# Train the
cd ..
echo "$CMD1"
echo "$CMD2"
CMD1=$(echo $JOB_CMD | cut -d ";" -f 1)
CMD2=$(echo $JOB_CMD | cut -d ";" -f 2)

python $CMD1
rm -rf $CMD2


# srun 
# python $JOB_CMD




