#!/bin/bash

# 2 * 5 * 5 * 5 = 250

#for LENGTH in 5 10 20
#do
#  for NSTART in 1.0 0.1
#  do
#    for LR in 10.0 3.0 1.0 0.3 0.1
#    do
#      for LAMBDA in 0.003 0.01 0.03 0.1 0.3
#      do
#        for SEED in $(seq 0 1 4) # $(seq 0 1 4)
#        do
rm hrpt_sst2.txt

# NEND=0.0001
# MODEL=gpt2-large # EleutherAI/gpt-j-6B,  EleutherAI/gpt-neo-2.7B,

# for LENGTH in 5 10 20
# do
#   for NSTART in 1.0 0.1
#   do
#     for LR in 10.0 3.0 1.0 0.3 0.1
#     do
#       for LAMBDA in 0.003 0.01 0.03 0.1 0.3
#       do
#         for SEED in $(seq 1000 1 1005) # $(seq 0 1 4)
#         do
NEND=0.0001
MODEL=gpt2-large # EleutherAI/gpt-j-6B,  EleutherAI/gpt-neo-2.7B,

for LENGTH in 5 10
do
  for NSTART in 1.0
  do
    for LR in 10.0 3.0 1.0 0.3
    do
      for LAMBDA in 0 0.0003 0.001 0.003 0.01 0.03 0.1 0.3
      do
        for SEED in $(seq 100 1 109) # $(seq 0 1 4)
        do
          if [ $(echo "$LAMBDA == 0" | bc -l) -eq 1 ]
          then 
            echo "hello $LAMBDA"
            NSTART=0.0000000001
            NEND=0.0000000001
          else
            NEND=0.0001
            NSTART=1.0
          fi

          MY_EXP_DIR="/private/home/swj0419/i-am-a-dog/openprompt-clone/swj_logging_dir2/sst2/gpt2-large/headprompt{$LENGTH}_LAMBDA{$LAMBDA}_LR{$LR}_NSTART{$NSTART}_NEND{$NEND}_SEED{$SEED}"
          mkdir -p $MY_EXP_DIR

          echo "my_experiments/cli.py \
              --config_yaml my_experiments/sst2_gpt2style_headprompt.yaml \
              --my_experiment_subdir $MY_EXP_DIR \
              --my_plm_name_or_path $MODEL \
              --project_prompt_embeds_interval 1 \
              --project_prompt_embeds_temp 1e-6 \
              --objective_prompt_labels_temp 1e-6 \
              --ppl_loss_lambda $LAMBDA \
              --prompt_lr $LR \
              --noise_start_value $NSTART \
              --noise_end_value $NEND \
              --my_global_seed $SEED \
              --prompt_num_tokens $LENGTH ; rm -rf $MY_EXP_DIR/checkpoints" >> hrpt_sst2.txt

  #        rm -rf $MY_EXP_DIR/checkpoints # remove the saved models
        done
      done
    done
  done
done

bash run_all_han1.sh hrpt_sst2.txt

