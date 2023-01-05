import sys
from typing import Dict
from omegaconf import OmegaConf,DictConfig
import wandb
import loguru

class Logger:
    def __init__(self, cfg:DictConfig):
        self.cfg = cfg
        self.logger = loguru.logger
        self.logger.remove()
        self.logger.add(f"{self.cfg.jobname}.log")

    def log_rank(self,data:Dict):
        log_str = "\t".join(f"{key}  {data[key]}" for key in data)
        self.logger.info(log_str)

    def log_master(self,data:Dict,if_wandb:bool=True):
        """
        only called in master process
        """
        log_str = "\t".join(f"{key}  {data[key]}" for key in data)
        self.logger.info(log_str)
        if self.cfg.debug != True and if_wandb:
            wandb.log(data)

    def login(self):
        """
        only login once in distributed training
        """
        self.logger.add(sys.stderr)
        if self.cfg.debug != True:
            wandb.login(key="a8c307987b041c73da9445e846682482ef2f526a")
            wandb.init(project="PromptedGeneration", entity="reign")
            wandb.config.update(self.cfg)

        self.logger.info("\nConfigs are:\n" + OmegaConf.to_yaml(self.cfg))

if __name__ == "__main__":
    mylogger = Logger()
    mylogger.log({"loss":0.1,"acc":0.9})
