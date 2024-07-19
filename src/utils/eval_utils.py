import copy
import torch
import time
import json
import evaluate
from tqdm import tqdm
import numpy as np
from transformers.data import DataCollatorForSeq2Seq
from utils.config_utils import get_dataloader_kwargs, generate_dataset_config
from dataset_srcs.samsum_dataset import get_preprocessed_samsum_for_rouge
from dataset_srcs.gsm8k_dataset import get_gsm8k_dataset, find_number


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

def em_for_gsm8k(train_config, model, tokenizer, wandb_run, full=False):
    dataset_config = generate_dataset_config(train_config)
    em_dataset = get_gsm8k_dataset(dataset_config, tokenizer, split="EM")
    em_config = copy.deepcopy(train_config)

    lengths = [len(d['input_ids']) for d in em_dataset]
    ids = np.argsort(lengths, kind='mergesort')

    em_config.batching_strategy = "padding"
    em_config.val_batch_size = 15 if train_config.few_shot == "none" else 5

    em_dl_kwargs = get_dataloader_kwargs(em_config, em_dataset, tokenizer, "val")
    em_dataloader = torch.utils.data.DataLoader(
        em_dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **em_dl_kwargs,
    )

    em_preds = []
    em_inputs = []
    em_labels = em_dataloader.dataset.labels

    for step, batch in enumerate(tqdm(em_dataloader,colour="green", desc="EM Epoch", dynamic_ncols=True)):
        key = 'input_ids'
        batch[key] = batch[key].to('cuda:0')

        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'], max_new_tokens=300)
        
        # preds = torch.argmax(outputs.logits, -1)
        em_preds.extend(
            tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
        )
        em_inputs.extend(
            tokenizer.batch_decode(batch['input_ids'].detach().cpu().numpy(), skip_special_tokens=True)
        )
        if not full:
            break
    
    num_correct = 0
    num_eval = len(em_preds)
    idx = 0
    em_dict = []
    for p in em_preds:
        em_ans = find_number(p)
        if em_ans == em_labels[ids[idx]]:
            num_correct += 1
        
        em_dict.append({
            'input ' : em_inputs[idx],
            'output' : p[len(em_inputs[idx]): ],
            'answer' : em_labels[ids[idx]],
            'extracted ans' : em_ans,
        })

        idx += 1
    
    t = time.time()
    with open(train_config.output_dir + f'/em_res{t}.json', 'w') as f :
        json.dump(em_dict, f, indent=4)
    
    if wandb_run:
        wandb_run.log({
            'EM_split' if not full else "EM" : num_correct / num_eval
        })

    print("EM: ", num_correct / num_eval)