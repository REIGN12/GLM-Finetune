import hydra
from omegaconf import DictConfig

from logger import Logger
from data import PGDataset,PGDataCollator
from model import PGModel

import torch
from torch import optim

from typing import Tuple,List
from torch import Tensor

import ignite.distributed as idist
from ignite.engine import Engine,Events
from ignite import metrics
from lrscheduler import build_lrscheduler
from ignite.handlers import Checkpoint, global_step_from_engine

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
        loss = model(batch).loss
        loss = loss / cfg.trainer.accumulate_steps # accumulate gradients
        loss.backward()
        if engine.state.iteration % cfg.trainer.accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    def test_evaluate_step(engine:Engine, batch:Tensor) -> Tuple[List[str],List[str]]:
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
    def rouge_output_transform(output:Tuple[List[str],List[str]]) -> Tuple[List[List[str]],List[List[List[str]]]]:
        res,labels = output
        res = [item.split() for item in res]
        labels = [[item.split()] for item in labels]
        return res,labels
    test_rouge = metrics.Rouge(variants=['L',1,2],output_transform=rouge_output_transform)
    test_rouge.attach(test_evaluator,"test_rouge")
    
    @trainer.on(Events.COMPLETED)
    def log_final_results(engine:Engine):
        log_data = { "epoch":engine.state.epoch, }
        # test evaluation of rouge is too slow, so we only evaluate it at the end
        test_evaluator.run(test_dataloader)
        test_metrics = test_evaluator.state.metrics
        log_data["test_rouge"] = test_metrics["test_rouge"]
        # log test evaluation
        logger.log_rank(log_data)
        if idist.get_rank() == 0:
            logger.log_master(log_data)
        return log_data
    
    # @trainer.on(Events.EPOCH_STARTED) # for debug
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_results(engine:Engine):
        log_data = { "epoch":engine.state.epoch, }
        # train evaluation
        train_metrics = engine.state.metrics
        log_data["train_loss"] = train_metrics["train_loss"]

        # qualitative study first
        model.eval()
        qualitative_num = cfg.trainer.qualitative_num
        qualitative_log_data = {}
        qualitative_rouge = metrics.Rouge() # calculate rouge for qualitative study
        with torch.no_grad():
            for batch in test_dataloader:
                prompts = batch.pop("prompts")
                labels = batch.pop("labels")
                batch.to(idist.device())
                res = model.generate(batch)
                res = res[:qualitative_num]
                labels = labels[:qualitative_num]
                # calculcate rouge for qulitative study examples
                qualitative_rouge.update(([item.split() for item in res],[[item.split()] for item in labels]))
                qualitative_log_data["qualitative_rouge"] = qualitative_rouge.compute()
                # log qualitative study examples
                for idx,(prompt,model_res,label) in enumerate(zip(prompts,res,labels)):
                    qualitative_log_data[f"qualitative_{idx}"] = {
                        "prompt":prompt,
                        "model_res":model_res,
                        "label":label,
                    }
                break
        # log qualitative study
        if idist.get_rank() == 0:
            logger.log_master(qualitative_log_data,if_wandb=False)

        # log train evaluation
        logger.log_rank(log_data)
        if idist.get_rank() == 0:
            logger.log_master(log_data)
        return log_data

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.trainer.log_interval))
    def log_step_results(engine:Engine):
        log_data = {
            "iteration":engine.state.iteration,
            "loss_per_step":engine.state.output,
            "lr":optimizer.param_groups[0]["lr"],
        }
        logger.log_rank(log_data)
        if idist.get_rank() == 0:
            logger.log_master(log_data)

    train_data_collator = PGDataCollator(cfg.data,"train")
    train_dataset = PGDataset(cfg.data,split="validation")
    if idist.get_rank() == 0:
        logger.log_master({
            "train dataset prompt_key":f"{train_dataset.prompt_key}"
        },if_wandb=False)

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

    # checkpointing; distributed is automatically handled
    to_save = {"model":model,"optimizer":optimizer,"trainer":trainer}
    checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.jobname}"
    checkpoint = Checkpoint(to_save,checkpoint_dir)#,global_step_from_engine(trainer))
    test_evaluator.add_event_handler(Events.STARTED,checkpoint)

    trainer.run(train_dataloader,max_epochs=cfg.trainer.epochs)

@hydra.main(version_base=None,config_path="config", config_name="basic")
def main(cfg: DictConfig) -> None:
    backend = cfg.distributed.backend
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(main_engine, cfg)
    return

# @hydra.main(version_base=None,config_path="config", config_name="basic")
# def main(cfg: DictConfig) -> None:
#     logger = Logger(cfg)

#     model = PGModel(cfg.model)
#     optimizer = optim.AdamW(model.parameters(),
#         lr=cfg.optimizer.lr,betas=(cfg.optimizer.beta1,cfg.optimizer.beta2),
#         weight_decay=cfg.optimizer.wd,
#     )

#     def train_step(engine:Engine, batch:Tensor) -> Tensor:
#         model.train()
#         optimizer.zero_grad()
#         loss = model(batch)
#         loss = model(batch).loss
#         loss.backward()
#         optimizer.step()
#         return loss.item()

#     trainer = Engine(train_step)
#     @trainer.on(Events.ITERATION_COMPLETED(every=cfg.trainer.log_interval))
#     def log_training(engine:Engine):
#         log_data = {
#             "epoch":engine.state.epoch,
#             "iteration":engine.state.iteration,
#             "loss":engine.state.output,
#         }
#         logger.log(log_data)


#     train_data_collator = PGDataCollator(cfg.data,"train")
#     train_dataset = PGDataset(cfg.data,split="train")
#     train_dataloader = DataLoader(
#         train_dataset,batch_size=cfg.trainer.batch, 
#         num_workers=cfg.trainer.num_workers,
#         pin_memory=cfg.trainer.pin_memory,
#         collate_fn=train_data_collator,
#         shuffle=False,drop_last=True,
#     )
#     trainer.run(train_dataloader,max_epochs=cfg.trainer.epochs)

#     return

#     test_data_collator = PGDataCollator(cfg.data,"test")
#     test_dataset = PGDataset(cfg.data,split="validation")
#     test_dataloader = DataLoader(
#         test_dataset,batch_size=cfg.trainer.batch,
#         collate_fn=test_data_collator,
#         shuffle=False,drop_last=False)

#     model = PGModel(cfg.model)
#     for cnt,data in enumerate(train_dataloader):
#         # __import__("ipdb").set_trace()
#         # data = data.to("cuda")
#         # if cnt < idx:
#         #     continue
#         # print(data)
#         res = model(data)
#         print(res.keys())
#         print(res.loss)
#         break
#     return

#     for idx,data in enumerate(test_dataloader):
#         labels = data.pop("labels")
#         res = model.generate(data)
#         print(res)
#         rouge = evaluate.load('rouge')
#         rouge_res = rouge.compute(predictions=res, references=labels)
#         print(rouge_res)
#         break

#         # if cnt >= idx:
#         #     break
#     # for data in train_dataloader:
#     #     print(data)
#     #     res = model(**data)
#     #     print(res)
#     #     print(res.loss)
#     #     print(res.keys())
#     #     break
#     # return


if __name__ == '__main__':
    main()
