# fluent_prompt
Code for Toward Human Readable Prompt Tuning


### Installing the dependencies
```bash
pip install -r requirements.txt
```


### Training
#### Write a config file
Write a yaml config file is a yaml file that specifies the hyperparameters and dataset for training. See `my_experiments/agnews_gpt2style_headprompt.yaml` for an example.


#### Run the training script
The training script is `my_experiments/cli.py`. It takes the following arguments: 
- `--config_yaml`: the path to the config yaml file
- `--my_experiment_subdir`: the path to the output directory
- `--my_plm_name_or_path`: the name or path of the pretrained language model. We use `gpt2-large` in our experiments.
- `--project_prompt_embeds_interval`: the interval of projecting the prompt embeddings to tokens.
- `--project_prompt_embeds_temp`: ？
- `--objective_prompt_labels_temp`: ？
- `--ppl_loss_lambda`: the weight of the perplexity loss.
- `--prompt_lr`: ？
- `--noise_start_value`: the start value of the noise.
- `--noise_end_value`: the end value of the noise.
- `--prompt_num_tokens`: the number of tokens in the prompt.

```bash
CONFIG_YAML=my_experiments/agnews_gpt2style_headprompt.yaml
OUTPUT_DIR=/path/to/output_dir
MODEL_NAME_OR_PATH=gpt2-large
python my_experiments/cli.py --config_yaml $CONFIG_YAML --my_experiment_subdir $OUTPUT_DIR --my_plm_name_or_path $MODEL_NAME_OR_PATH --project_prompt_embeds_interval 1 --project_prompt_embeds_temp 1e-6 --objective_prompt_labels_temp 1e-6 --ppl_loss_lambda 0 --prompt_lr 10.0 --noise_start_value 0.0000000001 --noise_end_value 0.0000000001 --my_global_seed 100 --prompt_num_tokens 5
```

