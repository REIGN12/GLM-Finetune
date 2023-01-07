from typing import Dict,Tuple,List
from torch import Tensor

import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig

# Support GLM, BART, T5
from transformers import AutoTokenizer
# Support commonsense_qa, multi_news
from datasets import load_dataset

# prompt support
from promptsource.templates import DatasetTemplates

class PCDataCollator:
    def __init__(self,datacollator_config: DictConfig):
        self.datacollator_config = datacollator_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.datacollator_config.tokenizer,trust_remote_code=True)
        self.collator = self.build_collator()
    def __call__(self, batch:List[Tuple[str,List[str],int]]) -> Dict[str,Tensor]:
        return self.collator(batch)
    def build_collator(self):
        if "glm" in self.datacollator_config.tokenizer:
            return self.glm_collator
        else:
            raise NotImplementedError("Not implemented yet")
    def glm_collator(self,batch:List[Tuple[str,List[str],int]]) -> Dict[str,Tensor]:
        prompts,choices_l,choice_ids = zip(*batch)
        prompts = self.tokenizer(prompts,return_tensors="pt",
                    padding=True,truncation=True,max_length=self.datacollator_config.max_length
        )
        res = self.tokenizer.build_inputs_for_multiple_choice(prompts,choices_l)
        labels = torch.tensor(choice_ids)
        res['labels'] = labels
        return res

class PCDataset(Dataset):
    def __init__(self,dataset_config: DictConfig,split:str):
        self.dataset_config = dataset_config
        self.dataset = load_dataset(*dataset_config.dataset.split("/"),split=split)
        self.prompt_key,self.prompter = self.build_prompter()
        self.adapter = self.build_adapter()

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index: int) -> Tuple[str,List[str],int]:
        data = self.dataset[index]
        prompt,choice = self.prompter.apply(data)
        choices_l = self.prompter.get_answer_choices_list(data)
        choice_id = choices_l.index(choice)
        prompt = prompt + "\n\n" + self.dataset_config.answer_prompt
        res = self.adapter(prompt,choices_l,choice_id)
        return res
    def build_adapter(self):
        if "glm" in self.dataset_config.tokenizer:
            return self.glm_adapter
        else:
            raise NotImplementedError("Not implemented yet")
    def glm_adapter(self,prompt:str,choices_l:List[str],choice_id:int) -> Tuple[str,List[str],int]:
        prompt += "[MASK]"
        return prompt,choices_l,choice_id
    def build_prompter(self):
        all_prompts = DatasetTemplates(self.dataset_config.dataset)
        # filter out those not original_task
        prompt_key = [name for name in all_prompts.all_template_names if all_prompts[name].metadata.original_task ]
        prompter = all_prompts[prompt_key[self.dataset_config.prompt_id]]
        return prompt_key,prompter

class PGDataCollator:
    def __init__(self,datacollator_config: DictConfig,split:str):
        self.datacollator_config = datacollator_config
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(self.datacollator_config.tokenizer,trust_remote_code=True)
        self.collator = self.build_collator()
    def build_collator(self):
        if "glm" in self.datacollator_config.tokenizer:
            if self.split == "train":
                return self.glm_train_collator
            else:
                return self.glm_test_collator
        else:
            raise NotImplementedError("Not implemented yet")
    def glm_train_collator(self,batch: List[Tuple[str,str]]) -> Dict[str,Tensor]:
        prompts,answers = [list(item) for item in zip(*batch)]
        res = self.tokenizer(prompts,padding=True,truncation=True,max_length=self.datacollator_config.max_length,return_tensors="pt")
        res = self.tokenizer.build_inputs_for_generation(res,targets=answers,max_gen_length=self.datacollator_config.max_gen_length)
        return res
    def glm_test_collator(self,batch: List[Tuple[str,str]]) -> Dict[str,Tensor]:
        prompts,answers = [list(item) for item in zip(*batch)]
        res = self.tokenizer(prompts,padding=True,truncation=True,max_length=self.datacollator_config.max_length,return_tensors="pt")
        res = self.tokenizer.build_inputs_for_generation(res,max_gen_length=self.datacollator_config.max_gen_length)
        res['labels'] = answers
        res['prompts'] = prompts
        return res

    def __call__(self, batch: List[Tuple[str,str]]) -> Dict[str, Tensor]:
        return self.collator(batch)

class PGDataset(Dataset):
    def __init__(self,dataset_config:DictConfig,split:str):
        """
        split = "train" or "validation" or "test"
        """
        self.dataset_config = dataset_config
        self.max_length = dataset_config.max_length
        self.max_gen_length = dataset_config.max_gen_length

        self.dataset = load_dataset(*dataset_config.dataset.split("/"),split=split)
        self.prompt_key,self.prompter = self.build_prompter()
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
        
    def glm_adapter(self,prompted_data:Tuple[str,str])->Tuple[str,str]:
        prompt,answer = prompted_data
        # add mask token
        prompt += "[MASK]"
        res = prompt,answer
        return res
        return res

    def build_prompter(self):
        all_prompts = DatasetTemplates(self.dataset_config.dataset)
        # filter out those not original_task
        prompt_key = [name for name in all_prompts.all_template_names if all_prompts[name].metadata.original_task ]
        prompter = all_prompts[prompt_key[self.dataset_config.prompt_id]]
        return prompt_key,prompter

    def __len__(self)->int:
        return len(self.dataset)
    def __getitem__(self, index:int)->Tuple[str,str]:
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

