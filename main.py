import hydra
from omegaconf import DictConfig

from logger import Logger
from data import PGDataset,PGDataCollator
from model import PGModel

import torch
from torch import optim

from torch import Tensor

import ignite.distributed as idist
from ignite.engine import Engine,Events
from ignite import metrics
from lrscheduler import build_lrscheduler

def main_engine(local_rank: int, cfg: DictConfig,**kwargs):
    # Setup logger
    logger = Logger(cfg)
    if idist.get_rank() == 0:
        logger.login()
    logger.log_rank({"rank":idist.get_rank(),"local_rank":local_rank,"world_size":idist.get_world_size()})

    # Setup model
    model = PGModel(cfg.model)
    model = idist.auto_model(model)
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(),
        lr=cfg.optimizer.lr,betas=(cfg.optimizer.beta1,cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.wd,
    )
    optimizer = idist.auto_optim(optimizer)

    def train_step(engine:Engine, batch:Tensor) -> Tensor:
        model.train()
        batch.to(idist.device())
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_evaluate_step(engine:Engine, batch:Tensor) -> Tensor:
        model.eval()
        prompts = batch.pop("prompts")
        labels = batch.pop("labels")
        with torch.no_grad():
            batch.to(idist.device())
            res = model.generate(batch)
            return res, labels

    # setup engines
    trainer = Engine(train_step)
    test_evaluator = Engine(test_evaluate_step)

    # metrics
    train_loss = metrics.Average()
    train_loss.attach(trainer,"train_loss")
    test_rouge = metrics.Rouge(output_transform=lambda output: [[item.split() for item in res] for res in output])
    test_rouge.attach(test_evaluator,"test_rouge")
    
    # @trainer.on(Events.EPOCH_STARTED) # for debug
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_results(engine:Engine):
        log_data = { "epoch":engine.state.epoch, }
        # train evaluation
        train_metrics = engine.state.metrics
        log_data["train_loss"] = train_metrics["train_loss"]

        # test evaluation
        test_evaluator.run(test_dataloader)
        test_metrics = test_evaluator.state.metrics
        log_data["test_rouge"] = test_metrics["test_rouge"]

        # qualitative study
        model.eval()
        qualitative_num = cfg.trainer.qualitative_num
        qualitative_log_data = {}
        with torch.no_grad():
            for batch in test_dataloader:
                prompts = batch.pop("prompts")
                labels = batch.pop("labels")
                batch.to(idist.device())
                res = model.generate(batch)
                res = res[:qualitative_num]
                labels = labels[:qualitative_num]
                for idx,(prompt,model_res,label) in enumerate(zip(prompts,res,labels)):
                    qualitative_log_data[f"qualitative_{idx}"] = {
                        "prompt":prompt,
                        "model_res":model_res,
                        "label":label,
                    }
                break

        logger.log_rank(log_data)
        logger.log_master(log_data)
        logger.log_master(qualitative_log_data,if_wandb=False)
        return log_data

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.trainer.log_interval))
    def log_step_results(engine:Engine):
        log_data = {
            "iteration":engine.state.iteration,
            "loss_per_step":engine.state.output,
            "lr":optimizer.param_groups[0]["lr"],
        }
        logger.log_rank(log_data)
        logger.log_master(log_data)

    train_data_collator = PGDataCollator(cfg.data,"train")
    train_dataset = PGDataset(cfg.data,split="train")
    train_dataloader = idist.auto_dataloader(
        train_dataset,batch_size=cfg.trainer.batch, 
        num_workers=cfg.trainer.num_workers,
        pin_memory=cfg.trainer.pin_memory,
        collate_fn=train_data_collator,
        shuffle=True,drop_last=True,
    )
    test_data_collator = PGDataCollator(cfg.data,"test")
    test_dataset = PGDataset(cfg.data,split="test")
    test_dataloader = idist.auto_dataloader(
        test_dataset,batch_size=cfg.trainer.batch,
        num_workers=cfg.trainer.num_workers,
        pin_memory=cfg.trainer.pin_memory,
        collate_fn=test_data_collator,
        shuffle=False,drop_last=False,
    )

    lrscheduler = build_lrscheduler(optimizer,cfg.trainer,len(train_dataset)//cfg.trainer.batch)
    trainer.add_event_handler(Events.ITERATION_STARTED,lrscheduler)

    trainer.run(train_dataloader,max_epochs=cfg.trainer.epochs)

@hydra.main(version_base=None,config_path="config", config_name="basic")
def main(cfg: DictConfig) -> None:
    backend = cfg.distributed.backend
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(main_engine, cfg)
    return

if __name__ == '__main__':
    main()
