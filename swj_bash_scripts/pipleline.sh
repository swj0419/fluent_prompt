#bash generate_job.sh
#bash run_all.sh hrpt.txt

TASK=amazon
echo $TASK
bash generate_job_$TASK.sh
bash run_all.sh hrpt_$TASK.txt