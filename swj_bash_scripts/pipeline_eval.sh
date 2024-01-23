#bash generate_job.sh
#bash run_all.sh hrpt.txt

#TASK=agnews
#echo $TASK
#bash generate_job_$TASK.sh
#bash run_all.sh hrpt_$TASK.txt

TASK=sst-2
echo $TASK
python generate_job_eval_agnews.py
bash run_all.sh hrpt_eval.txt