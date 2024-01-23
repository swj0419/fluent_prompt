#!/bin/bash


rm hrpt.txt

NEND=0.0001
LENGTH=5
NSTART=1.0
LR=0.1
LAMBDA=0.01
SEED=1000

INDEX=0
PROMPT='_disrespectful_film.\\n"'
ACC=0.885

PROMPT_PATH="/private/home/swj0419/i-am-a-dog/openprompt-clone/swj_logging_dir/sst2/eval/gpt2-large_prompt.csv"
while IFS=\; read -r INDEX PROMPT ACC PPL INPUT_PATH
do
    echo "$INDEX and $PROMPT"
done < $PROMPT_PATH



for MODEL in gpt2-xl EleutherAI/gpt-neo-2.7B # EleutherAI/gpt-j-6B,  EleutherAI/gpt-neo-2.7B,
do
#  MY_EXP_DIR="/private/home/swj0419/i-am-a-dog/openprompt-clone/swj_logging_dir/sst2/gpt2-large_soft/headprompt{10}_LAMBDA{0}_LR{0.3}_NSTART{1.0}_NEND{0.0001}_SEED{0}"
  EVAL_DIR="/private/home/swj0419/i-am-a-dog/openprompt-clone/swj_logging_dir/sst2/eval/$MODEL/headprompt_{$INDEX}_{$ACC}"
  echo "my_experiments/cli.py \
      --config_yaml my_experiments/sst2_gpt2style_headprompt_eval.yaml \
      --my_experiment_subdir $EVAL_DIR \
      --my_plm_name_or_path $MODEL \
      --project_prompt_embeds_interval 1 \
      --project_prompt_embeds_temp 1e-6 \
      --objective_prompt_labels_temp 1e-6 \
      --ppl_loss_lambda $LAMBDA \
      --prompt_lr $LR \
      --noise_start_value $NSTART \
      --noise_end_value $NEND \
      --my_global_seed $SEED \
      --prompt_num_tokens $LENGTH \
      --prompt_text $PROMPT"
#      >> hrpt_eval.txt
done


