from typing import Dict,Tuple
from torch import Tensor

from torch.utils.data import Dataset
from omegaconf import DictConfig

# Support GLM, BART, T5
from transformers import AutoTokenizer
# Support commonsense_qa, multi_news
from datasets import load_dataset

# prompt support
from promptsource.templates import DatasetTemplates

from transformers.tokenization_utils_base import BatchEncoding
import torch
class PGDataset(Dataset):
    def __init__(self,dataset_config:DictConfig,split:str):
        """
        split = "train" or "validation" or "test"
        """
        self.dataset_config = dataset_config
        self.max_length = dataset_config.max_length
        self.max_gen_length = dataset_config.max_gen_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.dataset_config.tokenizer,trust_remote_code=True)
        self.dataset = load_dataset(dataset_config.dataset,split=split)
        self.prompter = self.build_prompter()
        self.answer_prompt = dataset_config.answer_prompt
        self.adapter = self.build_adapter()


    
    def build_adapter(self):
        adapter_name = self.dataset_config.tokenizer
        if "glm" in adapter_name:
            adapter = self.glm_adapter
        elif "t5" in adapter_name:
            adapter = self.t5_adapter
        elif "bart" in adapter_name:
            adapter = self.bart_adapter
        else:
            raise NotImplementedError(f"Adapter {adapter_name} is not supported")
        return adapter
        
    def glm_adapter(self,prompted_data:Tuple[str,str])->Dict[str,Tensor]:
        def glm_build_inputs_for_generation(model_input:BatchEncoding, targets:str, max_gen_length:int)->Dict[str,Tensor]:
            """
            cp from https://huggingface.co/BAAI/glm-roberta-large/blob/main/tokenization_glm.py
            however rm line 168 in order to align the tensor shape
            """
            mask_ids = self.tokenizer.mask_token_ids
            input_ids = model_input.input_ids
            batch_size, seq_length = input_ids.shape[:2]
            position_id, block_position_id = list(range(seq_length)), [0 for _ in range(seq_length)]
            position_ids, block_position_ids = [], []
            labels = None
            if targets is not None:
                is_batched = isinstance(targets, (list, tuple))
                targets = self.tokenizer(targets, add_special_tokens=False, padding=False).input_ids
                if not is_batched:
                    targets = [targets]
                targets = [(target + [self.tokenizer.eop_token_id])[:max_gen_length] for target in targets]
                # max_gen_length = max(map(len, targets))
                targets = [[self.tokenizer.sop_token_id] + target + [-100] * (max_gen_length - len(target)) for target in targets]
                assert len(targets) == len(input_ids)
                targets = torch.tensor(targets, dtype=input_ids.dtype, device=input_ids.device)
                labels = torch.cat((input_ids.new_full((batch_size, seq_length), -100), targets[:, 1:]), dim=1)
            for i in range(batch_size):
                mask_positions = []
                for mask_id in mask_ids:
                    mask_positions += (input_ids[i] == mask_id).nonzero(as_tuple=True)[0].tolist()
                if not mask_positions:
                    raise ValueError("Cannot find mask token in the input")
                mask_positions.sort()
                mask_pos = mask_positions[0]
                position_ids.append(position_id + [mask_pos] * max_gen_length)
                block_position_ids.append(block_position_id + list(range(1, max_gen_length + 1)))
            position_ids = torch.tensor(position_ids, dtype=input_ids.dtype, device=input_ids.device)
            block_position_ids = torch.tensor(block_position_ids, dtype=input_ids.dtype, device=input_ids.device)
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)
            attention_mask = model_input.attention_mask
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length + max_gen_length, -1)
            generation_attention_mask = torch.cat([attention_mask.new_zeros((seq_length, max_gen_length)),
                                                   torch.tril(attention_mask.new_ones((max_gen_length, max_gen_length)))],
                                                  dim=0).unsqueeze(0).expand(batch_size, -1, -1)
            attention_mask = torch.cat((attention_mask, generation_attention_mask), dim=2)
            attention_mask = attention_mask.unsqueeze(1)
            if targets is None:
                input_ids = torch.cat((input_ids, input_ids.new_full((batch_size, 1), self.sop_token_id)), dim=-1)
            else:
                input_ids = torch.cat((input_ids, targets[:, :-1]), dim=1)
            batch = {"input_ids": input_ids, "position_ids": position_ids}
            if labels is None:
                batch["generation_attention_mask"] = attention_mask
            else:
                batch["attention_mask"] = attention_mask
                batch["labels"] = labels
            return BatchEncoding(batch)

        prompt,answer = prompted_data
        # add mask token
        prompt += "[MASK]"
        # tokenize
        res = self.tokenizer(prompt,padding='max_length',max_length=self.max_length,truncation=True, return_tensors="pt")
        res = glm_build_inputs_for_generation(res,targets=answer,max_gen_length=self.max_gen_length)
        return res
        return res

    def build_prompter(self):
        all_prompts = DatasetTemplates(self.dataset_config.dataset)
        # filter out those not original_task
        prompt_key = [name for name in all_prompts.all_template_names if all_prompts[name].metadata.original_task ]
        prompter = all_prompts[prompt_key[self.dataset_config.prompt_id]]
        return prompter

    def __len__(self)->int:
        return len(self.dataset)
    def __getitem__(self, index:int)->Dict[str,Tensor]:
        # TODO: format the data using prompt, add mask token based on model, padding based on max_lenght, then pass the tokenizer
        data = self.dataset[index]
        prompted_data = self.prompter.apply(data)
        prompted_data[0] = prompted_data[0] + "\n\n" + self.answer_prompt
        res = self.adapter(prompted_data)
        return res



if __name__ == "__main__":
    dataset = load_dataset("commonsense_qa")
    print(dataset.keys())
    dataset = load_dataset("multi_news")
    print(dataset.keys())

    # multi_news_prompts = DatasetTemplates("multi_news")
    # print(multi_news_prompts.all_template_names)

