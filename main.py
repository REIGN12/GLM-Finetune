import hydra
from omegaconf import DictConfig

from logger import Logger
from data import PGDataset,PGDataCollator
from model import PGModel

from torch import optim

from torch import Tensor

import ignite.distributed as idist
from ignite.engine import Engine,Events
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

    trainer = Engine(train_step)

    def get_log_data(engine:Engine):
        # TODO: add evaluation and average loss
        log_data = {
            "epoch":engine.state.epoch,
            "iteration":engine.state.iteration,
            "loss":engine.state.output,
            "lr":optimizer.param_groups[0]["lr"],
        }
        return log_data

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.trainer.log_interval))
    def log_training_rank(engine:Engine):
        log_data = get_log_data(engine)
        logger.log_rank(log_data)

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.trainer.log_interval))
    def log_training_master(engine:Engine):
        log_data = get_log_data(engine)
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
