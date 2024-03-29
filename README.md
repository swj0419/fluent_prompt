# fluent_prompt
Code for Toward Human Readable Prompt Tuning
* Our code structure borrows from [OpenPrompt](https://github.com/thunlp/OpenPrompt).


### Installing the dependencies
```bash
pip install -r requirements.txt
```


### Training
#### Write a config file
Write a yaml config file is a yaml file that specifies the hyperparameters and dataset for training. See `my_experiments/agnews_gpt2style_headprompt.yaml` for an example.


#### Run the training script and evaluate the model
The training script is `my_experiments/cli.py`. It takes the following arguments: 
- `--config_yaml`: the path to the config yaml file
- `--my_experiment_subdir`: the path to the output directory
- `--my_plm_name_or_path`: the name or path of the pretrained language model. We use `gpt2-large` in our experiments.
- `--project_prompt_embeds_interval`: the interval of projecting the prompt embeddings to tokens (i.e., existing embeddings in the embedding table).
- `--project_prompt_embeds_temp`: temp towards `0` means an one-hot projection, temp towards `+inf` means a uniform projection, see [here](https://github.com/swj0419/fluent_prompt/blob/494047e8a498abd709eb190c3cc04728004b1262/openprompt/prompts/soft_template.py#L134) for details.
- `--ppl_loss_lambda`: the weight of the perplexity loss.
- `--prompt_lr`: the prompt learning rate
- `--noise_start_value`: the start value of the noise.
- `--noise_end_value`: the end value of the noise.
- `--prompt_num_tokens`: the number of tokens in the prompt.

```bash
CONFIG_YAML=my_experiments/agnews_gpt2style_headprompt.yaml
OUTPUT_DIR=/path/to/output_dir
MODEL_NAME_OR_PATH=gpt2-large
python my_experiments/cli.py --config_yaml $CONFIG_YAML --my_experiment_subdir $OUTPUT_DIR --my_plm_name_or_path $MODEL_NAME_OR_PATH --project_prompt_embeds_interval 1 --project_prompt_embeds_temp 1e-6 --objective_prompt_labels_temp 1e-6 --ppl_loss_lambda 0 --prompt_lr 10.0 --noise_start_value 0.0000000001 --noise_end_value 0.0000000001 --my_global_seed 100 --prompt_num_tokens 5
```

