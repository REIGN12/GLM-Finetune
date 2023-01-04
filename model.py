import re

from torch import nn
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer

from typing import Dict,List
from omegaconf import DictConfig
from torch import Tensor

class PGModel(nn.Module):
    """
    Prompted Generation Model
    """
    def __init__(self,model_config:DictConfig):
        super(PGModel,self).__init__()
        self.model_config = model_config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_config.name,trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.name,trust_remote_code=True)
        self.generator = self.build_generator()
    def forward(self,data:Dict[str,Tensor]):
        res = self.model(**data)
        return res
    def generate(self,data:Dict[str,Tensor],**kwargs)->List[str]:
        return self.generator(data,**kwargs)
    def glm_generator(self,data:Dict[str,Tensor],**kwargs)->List[str]:
        res = self.model.generate(**data,
            max_new_tokens=self.model_config.max_gen_length,
            eos_token_id=self.tokenizer.eop_token_id,
        **kwargs)
        res = self.tokenizer.batch_decode(res.tolist())
        pattern = r"<\|startofpiece\|>(.*?)<\|endofpiece\|>"
        res = [re.search(pattern,txt,re.DOTALL).group(1) for txt in res]
        return res

    def build_generator(self):
        if "glm" in self.model_config.name:
            return self.glm_generator
        else:
            raise NotImplementedError("Not implemented yet")


if __name__ == "__main__":
    pass
