from typing import Dict
from torch import Tensor

from torch.utils.data import Dataset,DataLoader
from omegaconf import DictConfig

# Support GLM, BART, T5
from transformers import AutoTokenizer, T5Tokenizer, BartTokenizer
# Support commonsense_qa, multi_news
from datasets import load_dataset

# prompt support
from promptsource.templates import DatasetTemplates

class PGDataset(Dataset):
    def __init__(self,dataset_config:DictConfig):
        self.dataset_config = dataset_config
        self.tokenizer = self.build_tokenizer()
        self.dataset = load_dataset(dataset_config.dataset,split=dataset_config.split)
        self.prompter = self.build_prompter()


    def build_tokenizer(self):
        tokenizer_name = self.dataset_config.tokenizer
        if "glm" in tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,trust_remote_code=True)
        elif "t5" in tokenizer_name:
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        elif "bart" in tokenizer_name:
            tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        else:
            raise NotImplementedError(f"Tokenizer {tokenizer_name} is not supported")
        return tokenizer

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

        return self.dataset[index]



if __name__ == "__main__":
    dataset = load_dataset("commonsense_qa")
    print(dataset.keys())
    dataset = load_dataset("multi_news")
    print(dataset.keys())

    # multi_news_prompts = DatasetTemplates("multi_news")
    # print(multi_news_prompts.all_template_names)

