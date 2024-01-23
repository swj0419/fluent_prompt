# import os
import pandas as pd


def write_tofile():
    NEND=0.0001
    LENGTH=5
    NSTART=1.0
    LR=0.1
    LAMBDA=0.01
    SEED=1000

    INDEX=0
    PROMPT='_disrespectful_film.\\n"'
    ACC=0.885
    TASK = "agnews"

    PROMPT_PATH=f"/private/home/swj0419/i-am-a-dog/openprompt-clone/swj_logging_dir/{TASK}/eval/gpt2-large_prompt.csv"
    df = pd.read_csv(PROMPT_PATH)

    command_file = "hrpt_eval.txt"
    with open(command_file, "w") as f:
        for i, line in df.iterrows():
            if i > 20:
                break
            INDEX = i
            ACC = line["acc"]
            PPL = line["ppl"]
            PROMPT = line["prompt"]
            for MODEL in ["gpt2-xl", "EleutherAI/gpt-neo-2.7B"]:
                EVAL_DIR = f"generate_job_eval_agnews.py.." \
                           f"/{TASK}/eval/{MODEL}/headprompt_{round(INDEX, 2)}_ACC:{round(ACC, 2)}_PPL:{round(PPL, 2)}"
                line = f"my_experiments/cli.py \
                      --config_yaml my_experiments/sst2_gpt2style_headprompt_eval.yaml \
                      --my_experiment_subdir {EVAL_DIR} \
                      --my_plm_name_or_path {MODEL} \
                      --project_prompt_embeds_interval 1 \
                      --project_prompt_embeds_temp 1e-6 \
                      --objective_prompt_labels_temp 1e-6 \
                      --ppl_loss_lambda {LAMBDA} \
                      --prompt_lr {LR} \
                      --noise_start_value {NSTART} \
                      --noise_end_value {NEND} \
                      --my_global_seed {SEED} \
                      --prompt_num_tokens {LENGTH} \
                      --prompt_text {PROMPT}"
                # line = line.replace('\n', '\\n')
                f.write(line)
                f.write("\n")


if __name__ == "__main__":
    write_tofile()


