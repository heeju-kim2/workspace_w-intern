import copy
import torch
import evaluate
from tqdm import tqdm
import numpy as np
from transformers.data import DataCollatorForSeq2Seq
from utils.config_utils import get_dataloader_kwargs
from dataset_srcs.samsum_dataset import get_preprocessed_samsum_for_rouge


def rouge_for_samsum(train_config, model, tokenizer, accelerator, wandb_run):
    rouge_dataset, summaries = get_preprocessed_samsum_for_rouge(dataset_config=None, tokenizer=tokenizer)
    lengths = [len(d['input_ids']) for d in rouge_dataset]
    ids = np.argsort(lengths, kind='mergesort')

    metric = evaluate.load('rouge')

    rouge_dl_kwargs = get_dataloader_kwargs(train_config, rouge_dataset, tokenizer, "val")
    rouge_loader = torch.utils.data.DataLoader(
        rouge_dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **rouge_dl_kwargs,
    )

    # rouge_loader = torch.utils.data.DataLoader(rouge_dataset, pin_memory=True, batch_size=train_config.eval_batch_size, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer), shuffle=False)

    idx = 0
    for step, batch in enumerate(tqdm(rouge_loader, desc="calculating Rouge")):
        batch = {k :v.to(accelerator.device) for k, v in batch.items()}
        with torch.no_grad():
            # outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            outputs = model.generate(input_ids=batch['input_ids'].to('cuda:0'), max_new_tokens=100)

        # gather all inferences across chips
        accelerator.wait_for_everyone()

        input_prompts = tokenizer.batch_decode(batch['input_ids'].detach().cpu().numpy(), skip_special_tokens=True)
        output_prompts = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

        predictions = []
        references = []
        for i, p in zip(input_prompts, output_prompts):
            predictions.append(p[len(i):])
            references.append(summaries[ids[idx]])
            idx += 1

        metric.add_batch(
            predictions=predictions,
            references=references,
        )
        
        if step < 3:
            print("predictions: ", predictions[0])
            print("references: ", references[0])
        

    eval_metric = metric.compute()
    return eval_metric