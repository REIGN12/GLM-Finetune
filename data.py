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
        prompt,answer = prompted_data
        # add mask token
        prompt += "[MASK]"
        # tokenize
        res = self.tokenizer(prompt,padding='max_length',max_length=self.max_length,return_tensors="pt")
        res = self.tokenizer.build_inputs_for_generation(res,targets=answer,max_gen_length=self.max_gen_length)
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

