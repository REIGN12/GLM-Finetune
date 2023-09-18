# Dataset Name as Title

Author: Jingcheng Hu, hujc22@mails.tsinghua.edu.cn, https://reign12.github.io/

Student ID: 2022312848

## Task Description
### Dataset Statistics
### Task Introduction


## How to Train and Eval
### Dependency
You can activate your own conda env and run command
```bash
bash env_setup.sh cuda # If you are running on nvidia GPUs

bash env_setup.sh rocm # If you are running on amd GPUs
```
### Training and Evaluation
You can run `python main.py --help` or directly go to `./config/basic.yaml` to see all the supported configuration.

To run the distributed training, which will evaluate the results along the way; per step loss, per epoch loss and per epoch accuracy will be recorded:
```bash
torchrun --nproc_per_node <YOUR_GPU_NUM> main.py \
    task="pc" \ # this is a Prompted Choice task
    data.dataset="commonsense_qa" \
    model.name="BAAI/glm-roberta-large" \ # we also support bert-large-uncased, roberta-large
    data.prompt_id="2" \ # prompt_id of original_task=True prompt templates from promptsource; for the name of each prompt, you can refer to training log as you start the job, which will be like "train dataset prompt_key  ['answer_given_question_without_options', 'most_suitable_answer', 'question_answering', 'question_to_answer_index']"
    jobname=<ANY_NAME_YOU_LIKE> \
    debug=False \ # If you want to disable wandb, set debug=True; you can setup your wandb related var as env var, or just type it when the program need it; refer to logger.py for details
    optimizer.lr="1r-5" \ # no lr scaling will be done, this lr will be the final lr
    trainer.batch="32" \ # this is the total batch summed in all cards
    trainer.accumulate_steps="2" \ # we support gradient accumulate steps to have larger effective batch size
    trainer.epochs="10" trainer.warmup_epochs="1" # we use linear warmup and cosine decay
    # there are some more configs can be changed, please refer to ./config/basic.yaml for details and simply follow the pattern here
```

## Results
Report final performance and other methods.

## Reference
