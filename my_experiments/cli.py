import os
import sys
sys.path.append(".")

import argparse

from openprompt.trainer import BaseRunner, ClassificationRunner, GenerationRunner
from openprompt.lm_bff_trainer import LMBFFClassificationRunner
from openprompt.protoverb_trainer import ProtoVerbClassificationRunner
from re import template
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader
from openprompt.prompts import load_template, load_verbalizer, load_template_generator, load_verbalizer_generator
from openprompt.data_utils import FewShotSampler
from openprompt.utils.logging import config_experiment_dir, init_logger, logger
from openprompt.config import get_config, save_config_to_yaml
from openprompt.plms import load_plm_from_config
from openprompt.data_utils import load_dataset
from openprompt.utils.cuda import model_to_device
from pathlib import Path

import torch
import dill
from openprompt.config import get_user_config, add_cfg_to_argparser, update_cfg_with_argparser, check_config_conflicts

import os, shutil
import sys
sys.path.append(".")

import numpy as np
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from tensorboardX import SummaryWriter
from tqdm import tqdm
import warnings
from typing import Callable, Union, Dict
try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict
from openprompt.prompts import *
from openprompt.utils.metrics import classification_metrics, generation_metric
from transformers import  AdamW, get_linear_schedule_with_warmup
from transformers.optimization import  Adafactor, AdafactorSchedule


class MyClassificationRunner(BaseRunner):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer,
    or self-training can use other runner class.
    This class is specially implemented for classification.
    For generation task, though it can be integrated in this class
    via `task` option, we keep it as another class for simplicity.

    Args:
        model (:obj:`PromptForClassification`): One ``PromptForClassification`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """
    def __init__(self,
                 model: PromptForClassification,
                 config = None,
                 train_dataloader = None,
                 valid_dataloader = None,
                 test_dataloader = None,
                 loss_function = None,
                 id2label = None,
                 ):
        super().__init__(model = model,
                         config = config,
                         train_dataloader = train_dataloader,
                         valid_dataloader = valid_dataloader,
                         test_dataloader = test_dataloader,
                        )
        self.loss_function = loss_function if loss_function else self.configure_loss_function()
        self.id2label = id2label
        self.label_path_sep = config.dataset.label_path_sep

    def configure_loss_function(self,):
        r"""config the loss function if it's not passed."""
        if self.config.classification.loss_function == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif self.config.classification.loss_function == "nll_loss":
            return torch.nn.NLLLoss()
        else:
            raise NotImplementedError

    def observe_prompt_tokens(self):
        # soft_embed --> soft embeds projected to LUT
        _proj_prob_mat, proj_embed_mat = self.model.prompt_model.template.make_projected_embeds(
            projection_temperature=self.config.project_prompt_embeds_temp)
        therealprompt = self.model.prompt_model.template.convert_soft_embeds2_token(proj_embed_mat)
        logger.info(f"real prompt tokens: {therealprompt}")
        return therealprompt

    def set_noise_scale(self, start_value=None, end_value=None,\
        total_steps=None, prefix_steps=0, suffix_steps=0, schedule_option='geom'):
        self.sigma_full_list = None
        assert start_value is not None
        assert end_value is not None
        assert total_steps is not None
        assert prefix_steps + suffix_steps <= total_steps
        core_steps = total_steps - prefix_steps - suffix_steps
        if schedule_option == 'geom':
            sigma_list = list(np.geomspace(start_value, end_value, num=core_steps))
            self.sigma_full_list = [start_value] * prefix_steps + sigma_list + [end_value] * suffix_steps
        else:
            raise ValueError("check schedule_option")

    def get_noise_scale(self, global_step):
        return np.sqrt(2.0 * self.config.soft_template.optimize.lr * self.sigma_full_list[global_step])


    def training_epoch(self, epoch):
        self.model.train()
        self.model.zero_grad()
        total_loss = 0.0
        sum_loss, sum_task_loss, sum_ppl_loss = 0.0, 0.0, 0.0
        with tqdm(total=self.steps_per_epoch, desc=f"train epoch: {epoch}") as pbar:
            for batch_idx, batch in enumerate(self.train_dataloader):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()

                loss, task_loss, ppl_loss = self.training_step(batch, batch_idx)

                if self.config.train.gradient_accumulation_steps > 1:
                    loss = loss / self.config.train.gradient_accumulation_steps
                sum_loss += loss.item()

                sum_task_loss += task_loss.item() / self.config.train.gradient_accumulation_steps
                sum_ppl_loss += ppl_loss.item() / self.config.train.gradient_accumulation_steps

                loss.backward()
                
                if (batch_idx + 1) % self.config.train.gradient_accumulation_steps == 0:
                    pbar.set_postfix({'loss': sum_loss, 'task_loss': sum_task_loss, 'ppl_loss': sum_ppl_loss})
                    self.log('train/loss', sum_loss, self.global_step)
                    logger.info(f'sum loss: {sum_loss}')
                    # logger.info("{} {}".format(self.inner_model.template.soft_embeds.data.mean().item(),self.global_step))

                    if self.config.train.max_grad_norm > 0:
                        # raise ValueError("max_grad_norm is disabled for now to reflect the original langevin dynamics formula") # Han: can remove later if necessary
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.max_grad_norm)

                    for optimizer in self.optimizers:
                        optimizer.step()
                    for scheduler in self.schedulers:
                        scheduler.step()
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()

                    total_loss += sum_loss
                    sum_loss, sum_task_loss, sum_ppl_loss = 0., 0., 0.

                    # !! adding a scheduled noise [TODO] Han: since we are optimizing for the whole task, may have significantly more steps than Sachin's work; therefore, try applying noise only once after a few steps?
                    with torch.no_grad():
                        noise_scale = self.get_noise_scale(self.global_step)
                        noise = noise_scale * torch.normal(0, 1, size=self.model.prompt_model.template.soft_embeds.shape).to(self.model.prompt_model.template.soft_embeds.device)
                        self.model.prompt_model.template.soft_embeds.add_(noise)

                    # !! observe for each step (read-only, can skip intermittently)
                    self.observe_prompt_tokens()

                    # !! soft_embed --> vocab (hard reset to the LUT embedding space)
                    if (self.global_step+1) % self.config.project_prompt_embeds_interval == 0:
                        _proj_prob_mat, proj_embed_mat = self.model.prompt_model.template.make_projected_embeds(projection_temperature=self.config.project_prompt_embeds_temp)
                        with torch.no_grad():
                            self.model.prompt_model.template.soft_embeds.copy_(proj_embed_mat.clone().detach())


                    # eval
                    # print("self.global_step: ", self.global_step)
                    # print("batch_idx: ", batch_idx)
                    if (self.global_step % 500 == 0):
                        score = self.inference_epoch("validation")
                        copy = None
                        if self.best_score is None or (
                                (score - self.best_score) >= 0) == self.config.checkpoint.higher_better:
                            copy = 'best'
                            self.best_score = score
                        self.save_checkpoint('last', extra={"validation_metric": score}, copy=copy)


                    self.global_step += 1
                    pbar.update(1)

                if self.global_step >= self.num_training_steps:
                    self.observe_prompt_tokens()
                    # soft_embed --> vocab (copy)
                    _proj_prob_mat, proj_embed_mat = self.model.prompt_model.template.make_projected_embeds(
                        projection_temperature=self.config.project_prompt_embeds_temp)
                    with torch.no_grad():
                        self.model.prompt_model.template.soft_embeds.copy_(proj_embed_mat.clone().detach())
                    
                    logger.info(
                        f"Training epoch {epoch}, num_steps {self.global_step}, avg_loss: {total_loss / self.steps_per_epoch:.4f}, total_loss: {total_loss:.4f}")
                    return -1  # an indicator of stopping the training

        self.observe_prompt_tokens()
        # soft_embed --> vocab (copy)
        _proj_prob_mat, proj_embed_mat = self.model.prompt_model.template.make_projected_embeds(
            projection_temperature=self.config.project_prompt_embeds_temp)
        with torch.no_grad():
            self.model.prompt_model.template.soft_embeds.copy_(proj_embed_mat.clone().detach())

        logger.info(
            f"Training epoch {epoch}, num_steps {self.global_step},  avg_loss: {total_loss / self.steps_per_epoch:.4f}, total_loss: {total_loss:.4f}")
        return 1

    def inference_step(self, batch, batch_idx):
        label = batch.pop('label')
        logits = self.model(batch)
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist(), label.cpu().tolist()

    def inference_epoch_end(self, split, outputs):
        preds = []
        labels = []
        for pred, label in outputs:
            preds.extend(pred)
            labels.extend(label)

        self.save_results(split, {
            'preds': preds,
            'labels': labels,
        })

        metrics = OrderedDict()
        for metric_name in self.config.classification.metric:
            print("metric_name: ", metric_name)
            metric = classification_metrics(preds, labels, metric_name, id2label=self.id2label, label_path_sep=self.label_path_sep)
            metrics[metric_name] = metric

        return metrics

    # def ppl_loss_function(self, logits, label_prob): # logits: batch_size x 512 x num_vocab | label_prob (after softmax): 512 x num_vocab
    #     batch_size = logits.shape[0]
    #     input = torch.log_softmax(logits, dim=-1)[:, :-1, :]
    #     target = label_prob.repeat(batch_size, 1, 1) # repeat batch_size times: label_prob: batch_size x 512 x num_vocab
    #     target = target[:, 1:, :]
    #     cross_entropy_logits = -torch.sum(input * target, dim=-1) # batch_size x 512
    #     cross_entropy_logits_no_batch = torch.mean(cross_entropy_logits, dim=0) # 512
    #     return torch.mean(cross_entropy_logits_no_batch) # average over sequence length, can be changed to sum

    def embedding_ppl_loss_function(self, plm_last_h, soft_prompts, lut_embeds):
        batch_size = plm_last_h.size(0)
        hidden_dim = plm_last_h.size(2)
        h = plm_last_h[:, :-1, :].contiguous().view(-1, hidden_dim) # BS * (N_prompts-1), h
        s = soft_prompts[1:, :].repeat(batch_size, 1, 1).view(-1, hidden_dim) # BS * (N_prompts-1), h
        hs = torch.sum(h * s, dim=1) # BS * (N_prompts-1)

        e = lut_embeds.view(-1, hidden_dim) # V * h
        he = torch.matmul(h, e.T) # BS * (N_prompts-1), V

        logp_elem = hs - torch.logsumexp(he, dim=1) # BS * (N_prompts-1)
        return -torch.mean(logp_elem)

    def training_step(self, batch, batch_idx):
        # [TODO] Han: no projection when calculating loss! (1) take the next embedding representation (2) find the most similar embedding in LUT (using straight-through operator) (3) calculate loss according to Sachin's Eq 5; drawback: input and output embedding weights in the model must be shared

        logits = self.model(batch) # size of logits is batch_size x num_of_verbalizers, they are log-probabilities, can be passed to another CE loss without problem (since multiple logsoftmax layers are equivalent to one)
        task_loss = self.loss_function(logits, batch['label'])

        # ppl loss (embedding method according to Sachin's Eq 5)
        soft_prompts = self.model.prompt_model.template.soft_embeds # N_prompt x h
        real_plm_last_h = self.model.prompt_model.plm_last_hidden_states
        plm_last_h = real_plm_last_h[:, :soft_prompts.size(0),: ] # BS x N_prompt x h
        lut_embeds = self.model.prompt_model.template.raw_embedding.weight # V x h, require_grads should be false
        ppl_loss = self.embedding_ppl_loss_function(plm_last_h, soft_prompts, lut_embeds)

        loss = (1 - self.config.ppl_loss_lambda) * task_loss + self.config.ppl_loss_lambda * ppl_loss

        return loss, task_loss, ppl_loss

    def on_fit_start(self):
        """Some initialization works"""
        self.prompt_initialize()
        self.set_noise_scale(start_value=self.config.noise_start_value, end_value=self.config.noise_end_value, total_steps=self.num_training_steps)

    def prompt_initialize(self):
        verbalizer_config = self.config[self.config.verbalizer]
        template_config = self.config[self.config.template]
        if not hasattr(self.inner_model.verbalizer, "optimize_to_initialize" ) and \
            not hasattr(self.inner_model.template, "optimize_to_initialize" ):
            return None
        if hasattr(verbalizer_config, "init_using_split"):
            using_split = verbalizer_config.init_using_split
        elif hasattr(template_config, "init_using_split"):
            using_split = template_config.init_using_split
        else:
            using_split = "valid"

        if using_split == "train":
            dataloader = self.train_dataloader
        elif using_split == "valid":
            dataloader = self.valid_dataloader
        else:
            raise NotImplementedError

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Init_using_{}".format(using_split)):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
                logits = self.model(batch)
            if hasattr(self.inner_model.verbalizer, "optimize_to_initialize" ):
                self.inner_model.verbalizer.optimize_to_initialize()
            if hasattr(self.inner_model.template, "optimize_to_initialize" ):
                self.inner_model.template.optimize_to_initialize()

        self.wrap_model()

    def run(self, ckpt: Optional[str] = None) -> dict:
        self.fit(ckpt)
        self.test(ckpt=None if self.clean else 'best')
        self.observe_prompt_tokens()
        return 0


def my_get_config():
    parser = argparse.ArgumentParser("Global Config Argument Parser", allow_abbrev=False)
    parser.add_argument("--config_yaml", required=True, type=str, help='the configuration file for this experiment.')
    parser.add_argument("--resume", type=str, help='a specified logging path to resume training.\
           It will fall back to run from initialization if no lastest checkpoint are found.')
    parser.add_argument("--test", type=str, help='a specified logging path to test')

    parser.add_argument("--my_experiment_subdir", type=str, default='PLACEHOLDER', help='n/a')
    parser.add_argument("--my_global_seed", type=int, default=None, help='n/a')
    parser.add_argument("--my_plm_name_or_path", type=str, default=None, help='n/a')
    parser.add_argument("--my_interpretation_split", type=str, default=None, help='n/a')

    parser.add_argument("--my_resuming_prompt_model", type=str, default=None, help='n/a')
    parser.add_argument("--my_resuming_language_model", type=str, default=None, help='n/a')

    parser.add_argument("--init_blank_language_model", type=str, default="no", help='n/a')

    parser.add_argument("--ppl_loss_lambda", type=float, default=0.5, help='')
    parser.add_argument("--project_prompt_embeds_interval", type=int, default=30, help='')
    parser.add_argument("--project_prompt_embeds_temp", type=float, default=0.001, help='')
    parser.add_argument("--objective_prompt_labels_temp", type=float, default=0.01, help='')

    parser.add_argument("--prompt_lr", type=float, default=0.01, help='')

    parser.add_argument("--noise_start_value", type=float, default=5, help='')
    parser.add_argument("--noise_end_value", type=float, default=0.05, help='')

    parser.add_argument("--prompt_num_tokens", type=int, default=5, help='')
    parser.add_argument("--prompt_text", type=str, default="", help='') #  nargs='+'

    args, _ = parser.parse_known_args()
    config = get_user_config(args.config_yaml)

    add_cfg_to_argparser(config, parser)
    args = parser.parse_args()

    update_cfg_with_argparser(config, args)
    check_config_conflicts(config)

    # update config manually
    config.logging.unique_string = args.my_experiment_subdir
    config.reproduce.seed = args.my_global_seed
    config.plm.model_path = args.my_plm_name_or_path
    config.soft_template.num_tokens = args.prompt_num_tokens

    # prompt text
    # config.soft_template.prompt_text = " ".join(args.prompt_text)
    config.soft_template.prompt_text = args.prompt_text.replace("_", " ")
    print("text_prompt: ", config.soft_template.prompt_text)
    logger.info(f"text_prompt: {config.soft_template.prompt_text}")
    config.soft_template.optimize.lr = args.prompt_lr

    if 'sampling_from_train' in config:
        config.sampling_from_train.seed = [args.my_global_seed]
    if args.my_interpretation_split:
        config.interpretation.probing_split = args.my_interpretation_split

    if args.my_resuming_prompt_model:
        config.resuming.my_resuming_prompt_model = args.my_resuming_prompt_model
    if args.my_resuming_language_model:
        config.resuming.my_resuming_language_model = args.my_resuming_language_model

    if args.init_blank_language_model == "yes":
        config.plm.load_blank_model = True
    elif args.init_blank_language_model == "no":
        config.plm.load_blank_model = False
    else:
        raise ValueError("check args.init_blank_language_model")

    config.ppl_loss_lambda = args.ppl_loss_lambda
    config.project_prompt_embeds_interval = args.project_prompt_embeds_interval
    config.project_prompt_embeds_temp = args.project_prompt_embeds_temp
    config.objective_prompt_labels_temp = args.objective_prompt_labels_temp

    config.noise_start_value = args.noise_start_value
    config.noise_end_value = args.noise_end_value

    return config, args

def build_dataloader(dataset, template, tokenizer,tokenizer_wrapper_class, config, split):
    dataloader = PromptDataLoader(
        dataset = dataset,
        template = template,
        tokenizer = tokenizer,
        tokenizer_wrapper_class=tokenizer_wrapper_class,
        batch_size = config[split].batch_size,
        shuffle = config[split].shuffle_data,
        teacher_forcing = config[split].teacher_forcing if hasattr(config[split],'teacher_forcing') else None,
        predict_eos_token = True if config.task == "generation" else False,
        **config.dataloader
    )
    return dataloader



def main():
    config, args = my_get_config()
    # exit()
    # init logger, create log dir and set log level, etc.

    # create output dir
    Path(args.my_experiment_subdir).mkdir(parents=True, exist_ok=True)

    if args.resume and args.test:
        raise Exception("cannot use flag --resume and --test together")
    if args.resume or args.test:
        config.logging.path = EXP_PATH = args.resume or args.test
    else:
        EXP_PATH = config_experiment_dir(config)
        init_logger(os.path.join(EXP_PATH, "log.txt"), config.logging.file_level, config.logging.console_level)
        # save config to the logger directory
        save_config_to_yaml(config)


    # load dataset. The valid_dataset can be None
    # train_dataset, valid_dataset, test_dataset, Processor = load_dataset(config, test = args.test is not None or config.learning_setting == 'zero_shot')
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(config, test = False) # Han: 221011, force loading all
    if valid_dataset is not None:
        print("valid_dataset: ", len(valid_dataset))
    print("test_data: ", len(test_dataset))
    # main
    if config.learning_setting == 'full':
        res = trainer(
            EXP_PATH,
            config,
            Processor,
            resume = args.resume,
            test = args.test,
            train_dataset = train_dataset,
            valid_dataset = valid_dataset, # Han: note here, weijia note here
            test_dataset = valid_dataset,
        )
    elif config.learning_setting == 'few_shot':
        if config.few_shot.few_shot_sampling is None:
            raise ValueError("use few_shot setting but config.few_shot.few_shot_sampling is not specified")
        seeds = config.sampling_from_train.seed
        res = 0
        for seed in seeds:
            if not args.test:
                sampler = FewShotSampler(
                    num_examples_per_label = config.sampling_from_train.num_examples_per_label,
                    also_sample_dev = config.sampling_from_train.also_sample_dev,
                    num_examples_per_label_dev = config.sampling_from_train.num_examples_per_label_dev
                )
                train_sampled_dataset, valid_sampled_dataset = sampler(
                    train_dataset = train_dataset,
                    valid_dataset = valid_dataset,
                    seed = seed
                )
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    Processor,
                    resume = args.resume,
                    test = args.test,
                    train_dataset = train_sampled_dataset,
                    valid_dataset = test_dataset, # Han: note here
                    test_dataset = test_dataset,
                )
            else:
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    Processor,
                    test = args.test,
                    test_dataset = test_dataset,
                )
            res += result
        res /= len(seeds)
    elif config.learning_setting == 'zero_shot':
        res = trainer(
            EXP_PATH,
            config,
            Processor,
            zero = True,
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            test_dataset = test_dataset, # Han: can consider changing to valid_dataset
        )

def trainer(EXP_PATH, config, Processor, train_dataset = None, valid_dataset = None, test_dataset = None, resume = None, test = None, zero = False):
    if not os.path.exists(EXP_PATH):
        os.mkdir(EXP_PATH)
    config.logging.path = EXP_PATH
    # set seed
    set_seed(config.reproduce.seed)

    # load the pretrained models, its model, tokenizer, and config.
    plm_model, plm_tokenizer, plm_config, plm_wrapper_class = load_plm_from_config(config)



    # define template and verbalizer
    if config.task == "classification":
        # define prompt
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        verbalizer = load_verbalizer(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config, classes=Processor.labels)
        # load prompt’s pipeline model
        prompt_model = PromptForClassification(plm_model, template, verbalizer, freeze_plm = config.plm.optimize.freeze_para)

    elif config.task == "generation":
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        prompt_model = PromptForGeneration(plm_model, template, freeze_plm = config.plm.optimize.freeze_para, gen_config = config.generation)
    else:
        raise NotImplementedError(f"config.task {config.task} is not implemented yet. Only classification and generation are supported.")

    # process data and get data_loader
    train_dataloader = build_dataloader(train_dataset, template, plm_tokenizer, plm_wrapper_class, config, "train") if train_dataset else None
    valid_dataloader = build_dataloader(valid_dataset, template, plm_tokenizer, plm_wrapper_class, config, "dev") if valid_dataset else None
    test_dataloader = build_dataloader(test_dataset, template, plm_tokenizer, plm_wrapper_class, config, "test") if test_dataset else None

    if config.task == "classification":
        if config.classification.auto_t or config.classification.auto_v:
            runner = LMBFFClassificationRunner(train_dataset = train_dataset,
                                                valid_dataset = valid_dataset,
                                                test_dataset = test_dataset,
                                                template=template,
                                                verbalizer=verbalizer,
                                                config = config
                                                )
        elif config.verbalizer == "proto_verbalizer":
            runner = ProtoVerbClassificationRunner(model = prompt_model,
                                    train_dataloader = train_dataloader,
                                    valid_dataloader = valid_dataloader,
                                    test_dataloader = test_dataloader,
                                    id2label = Processor.id2label,
                                    config = config
            )
        else: # Han: make a similar version for generation mode later
            runner = MyClassificationRunner(model = prompt_model,
                                    train_dataloader = train_dataloader,
                                    valid_dataloader = valid_dataloader,
                                    test_dataloader = test_dataloader,
                                    id2label = Processor.id2label,
                                    config = config
            )
    elif config.task == "generation":
        runner = GenerationRunner(
            model = prompt_model,
            train_dataloader = train_dataloader,
            valid_dataloader = valid_dataloader,
            test_dataloader = test_dataloader,
            config = config
        )

    if zero:
        res = runner.test()
    elif test:
        res = runner.test(ckpt = 'best')
    elif resume:
        res = runner.run(ckpt = 'last')
    else:
        res = runner.run()
    return res


if __name__ == "__main__":
    main()
