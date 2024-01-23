
import os

from torch.nn.parameter import Parameter
from openprompt.utils.logging import logger



from openprompt.data_utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer

import torch
from torch import nn

class SoftTemplate(Template):
    r"""This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens.
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take
    the first n_tokens similar to their implementation).
    """
    registered_inputflag_names = ["loss_ids", "shortenable_ids"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 soft_embeds: Optional[torch.FloatTensor] = None,
                 num_tokens: int=20,
                 initialize_from_vocab: Optional[bool] = True,
                 random_range: Optional[float] = 0.5,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                 prompt_position: str = 'head',
                 prompt_text: Optional[str] = None,
                ):
        super().__init__(tokenizer=tokenizer,
                         placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.raw_embedding.requires_grad_(False)
        self.model_is_encoder_decoder = model.config.is_encoder_decoder
        self.random_range = random_range
        self.num_tokens = num_tokens
        self.initialize_from_vocab = initialize_from_vocab

        self.text = text
        self.mode = "l2"

        # self.default_text1 = {"placeholder<text_a> <mask>"
        # self.default_text2 = "<text_a> <text_b> <mask>".split()

        self.model = model
        self.projected_probability_matrix = None
        self.projected_embeds_matrix = None

        if soft_embeds is not None:
            self.soft_embeds = soft_embeds
            self.num_tokens = len(soft_embeds)
        else:
            if self.num_tokens>0:
                self.generate_parameters(prompt_text)

        self.prompt_position = prompt_position # Han: head or tail, adding to both would require more work

    def convert_soft_embeds2_token(self, _input_soft_embeds):
        input_soft_embeds = _input_soft_embeds.clone().detach()
        with torch.no_grad():
            if self.mode == 'l2':
                proj_score_mat = input_soft_embeds.view(self.num_tokens, 1, -1) - torch.unsqueeze(self.model.get_input_embeddings().weight, 0)  # shape num_tokens, num_vocab, n_dim
                proj_score_mat = - torch.linalg.norm(proj_score_mat, ord=2, dim=-1)
            elif self.mode == 'dot':
                proj_score_mat = torch.matmul(input_soft_embeds, torch.transpose(self.model.get_input_embeddings().weight, 0, 1)) #num_tokens, num_vocab
            real_token_ids = torch.argmax(proj_score_mat, dim=-1)
        real_tokens = self.tokenizer.convert_ids_to_tokens(real_token_ids)
        real_string = self.tokenizer.convert_tokens_to_string(real_tokens)
        # print("real_string", real_string)
        return real_string

    def on_text_set(self):
        self.text = self.parse_text(self.text)


    def wrap_one_example(self, example) -> List[Dict]:  #TODO this automatic generated template may not be able to process diverse data format.
        if self.text is None:
            logger.warning("You didn't provide text template for softprompt. Using default template, is this intended?")
            if example.text_b is None:
                self.text = self.default_text1
            else:
                self.text = self.default_text2
        return super().wrap_one_example(example)


    def generate_parameters(self, prompt_text=None) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        # swj
        # if self.initialize_from_vocab:
        #     soft_embeds = self.raw_embedding.weight[:self.num_tokens].clone().detach()
        # else:
        #     soft_embeds = torch.FloatTensor(self.num_tokens, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)

        #### Han: manually comment out unused ones
        # # design choice 1 -- from random embedding LUT
        # soft_embeds = self.raw_embedding.weight[:self.num_tokens].clone().detach()
        # design choice 2 -- from uniform distribution
        # soft_embeds = torch.FloatTensor(self.num_tokens, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)
        # # design choice 3 -- from natural language sentence
        # manual_ids = torch.LongTensor(self.tokenizer.encode(" apple is a dog dog" * self.num_tokens)[:self.num_tokens]).to(self.model.device)
        # soft_embeds = self.raw_embedding.weight[manual_ids].clone().detach()
        print("prompt_text: ", prompt_text)
        if prompt_text == "":
            soft_embeds = torch.FloatTensor(self.num_tokens, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)
        else:
            manual_ids = self.tokenizer.encode(prompt_text)
            print("prompt text ids: ", manual_ids)
            # assert len(manual_ids) == self.num_tokens # Han: can add [:self.num_tokens] later, but omit for now for safety
            if len(manual_ids) != self.num_tokens:
                print("The prompt your entered doesn't match the number of soft tokens specified. Are you sure you want to continue?")
                breakpoint()
            manual_ids = torch.LongTensor(manual_ids).to(self.model.device)
            soft_embeds = self.raw_embedding.weight[manual_ids].clone().detach()

        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True)
        # swj to be deleted
        print("initial prompts", self.convert_soft_embeds2_token(soft_embeds))

    def make_projected_embeds(self, projection_temperature): # temp -> 0: one-hot, temp -> +inf: uniform
        if self.mode == "l2": # (num_tokens, 1, num_embed) - (num_vocab, num_embed) =
            proj_score_mat = self.soft_embeds.view(self.num_tokens, 1, -1) - torch.unsqueeze(self.model.get_input_embeddings().weight, 0) # shape num_tokens, num_vocab, n_dim
            proj_score_mat = - torch.linalg.norm(proj_score_mat, ord=2, dim=-1) # this number is negative and can be large
        elif self.mode == "dot":
            proj_score_mat = torch.matmul(self.soft_embeds, torch.transpose(self.model.get_input_embeddings().weight, 0, 1))
        self.projected_probability_matrix = nn.functional.softmax(proj_score_mat / projection_temperature, dim=-1)
        self.projected_embeds_matrix = torch.matmul(self.projected_probability_matrix, self.model.get_input_embeddings().weight) # input_embed: [num_vocab, n_dim]
        # return self.projected_probability_matrix.clone().detach(), self.projected_embeds_matrix.clone().detach()
        return self.projected_probability_matrix, self.projected_embeds_matrix


    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        batch_size = inputs_embeds.size(0)
        if self.num_tokens>0:
            soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
            if self.prompt_position == "head":
                inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1)
            elif self.prompt_position == "tail": # Han: change here for tailing prompt
                # first cat each batch element as [t2[1,2,...,n-1], t1, t2[n, n+1, ...]], where location n is the mask token for the task loss
                # then cat the batch elements back
                bread_tensor = inputs_embeds
                meat_tensor = soft_embeds
                sandwich_tensor = torch.cat([torch.cat([t2[:batch['input_ids_len'][i]-1], t1, t2[batch['input_ids_len'][i]-1:]], 0).unsqueeze(0) for i, (t1, t2) in enumerate(zip(meat_tensor, bread_tensor))], 0)
                inputs_embeds = sandwich_tensor
            else:
                raise ValueError("check prompt_position")

        batch['input_ids_copy'] = batch['input_ids']
        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens>0:
            am = batch['attention_mask']
            if self.prompt_position == "head":
                batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_tokens), dtype = am.dtype,device=am.device), am], dim=-1)
            elif self.prompt_position == "tail": # Han: change here for tailing prompt
                bread_tensor = am
                meat_tensor = torch.ones((batch_size,self.num_tokens), dtype = am.dtype,device=am.device)
                sandwich_tensor = torch.cat([torch.cat([t2[:batch['input_ids_len'][i]-1], t1, t2[batch['input_ids_len'][i]-1:]], 0).unsqueeze(0) for i, (t1, t2) in enumerate(zip(meat_tensor, bread_tensor))], 0)
                batch['attention_mask'] = sandwich_tensor
            else:
                raise ValueError("check prompt_position")
        return batch


    def post_processing_outputs(self, outputs: torch.Tensor):
        r"""Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        if not self.model_is_encoder_decoder:
            if self.prompt_position == "head":
                outputs.logits = outputs.logits[:, self.num_tokens:,: ]
            elif self.prompt_position == "tail": # Han: change here for tailing prompt
                # Han: A VERY BAD HACK: this will work in our case, since later we only care about the last logit (where batch['loss_ids'] > 0, our soft prompt won't affect it)
                outputs.logits = outputs.logits[:, self.num_tokens:,: ]
                # # Han: below won't work for now, since we didn't pass in `batch`
                # outputs.logits = torch.cat([torch.cat([t[:batch['input_ids_len'][i]-1], t[batch['input_ids_len'][i]-1+self.num_tokens:]], 0) for i, t in enumerate(outputs.logits)], 0)
            else:
                raise ValueError("check prompt_position")
        return outputs
