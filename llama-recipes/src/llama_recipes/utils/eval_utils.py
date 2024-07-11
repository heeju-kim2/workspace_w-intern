import copy
import torch
from tqdm import tqdm
import evaluate
import numpy as np
from torch.nn import CrossEntropyLoss
from llama_recipes.datasets import get_gsm8k_dataset
from llama_recipes.utils.config_utils import get_dataloader_kwargs
from llama_recipes.datasets.samsum_dataset import prepare_samsum_data
from llama_recipes.datasets.gsm8k.dataset import find_number
from llama_recipes.datasets.hellaswag_dataset import HellaDataset
from transformers.data import DataCollatorForSeq2Seq

def get_rouge(eval_preds, test_labels):
    """
    Calculate the rouge score
    """
    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=eval_preds, references=test_labels)

    return scores

def rouge_for_samsum(train_config, model, tokenizer, wandb_run):
    rouge_dataset, instructions, summaries = prepare_samsum_data(tokenizer)
    rouge_labels = []
    rouge_results = []
    rouge_inputs = []
    rouge_orgn = []

    lengths = [len(d['input_ids']) for d in rouge_dataset]
    ids = np.argsort(lengths, kind='mergesort')

    # packing 말고 padding strategy로 batch 진행
    # data -> sampler.py에서 sampler import
    # sampler에서 length 순서로 정렬 후 batch 생성 -> 데이터 순서 달라짐
    # 정렬한 순서 = ids => instructions, summaries ids로 순서 일치
    rouge_config = copy.deepcopy(train_config)
    rouge_config.batching_strategy = "padding"
    rouge_config.val_batch_size = 15

    rouge_dl_kwargs = get_dataloader_kwargs(rouge_config, rouge_dataset, tokenizer, "val")

    rouge_loader = torch.utils.data.DataLoader(
        rouge_dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **rouge_dl_kwargs,
    )
    idx = 0
    for step, batch in enumerate(tqdm(rouge_loader, desc="calculating Rouge")):
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'].to('cuda:0'), max_new_tokens=100)

            rouge_result = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )

            rouge_input = tokenizer.batch_decode(
                batch['input_ids'].detach().cpu().numpy(), skip_special_tokens=True
            )

            for r in rouge_result:
                rouge_results.append(r[len(instructions[ids[idx]]): ])
                rouge_labels.append(summaries[ids[idx]])
                rouge_orgn.append(r)
                idx += 1
            
            rouge_inputs.extend(rouge_input)
    
    if len(rouge_labels) != len(rouge_results):
        print("In train_utils rouge_results and rouge_labels length mismatched")
    rouge = get_rouge(rouge_results, rouge_labels[:len(rouge_results)])
    print(rouge)
    rouge_dict = []
    for r, l, i, o in zip(rouge_results, rouge_labels, rouge_inputs, rouge_orgn):
        rouge_dict.append({
            'input' : i,
            'output' : r,
            'label' : l,
            'orgn' : o,
        })
    
    import json
    with open(train_config.output_dir + f'/rouge_res.json', 'w') as f :
        json.dump(rouge_dict, f, indent=4)

    if wandb_run:
        wandb_run.log(rouge)


def em_for_gsm8k(train_config, dataset_config, model, tokenizer, wandb_run, full=False):
    em_dataset = get_gsm8k_dataset(dataset_config, tokenizer, split="EM")
    em_config = copy.deepcopy(train_config)

    lengths = [len(d['input_ids']) for d in em_dataset]
    ids = np.argsort(lengths, kind='mergesort')

    em_config.batching_strategy = "padding"
    em_config.val_batch_size = 15 if dataset_config.few_shot == "none" else 5

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
    
    import json
    import time
    t = time.time()
    with open(train_config.output_dir + f'/em_res{t}.json', 'w') as f :
        json.dump(em_dict, f, indent=4)
    
    if wandb_run:
        wandb_run.log({
            'EM_split' if not full else "EM" : num_correct / num_eval
        })

    print("EM: ", num_correct / num_eval)


def acc_for_hella(train_config, model, tokenizer, wandb_run):
    acc_dataset = HellaDataset(tokenizer)
    acc_batch_size = 16 ## 꼭 4의 배수로 설정해주세요
    acc_loader = torch.utils.data.DataLoader(acc_dataset, pin_memory=True, batch_size=acc_batch_size, collate_fn=DataCollatorForSeq2Seq(tokenizer))
    num_correct = 0
    num_eval = 0

    pbar = tqdm(colour="blue", desc=f"Evaluating Acc", total=len(acc_loader), dynamic_ncols=True)
    for step, batch in enumerate(acc_loader):
        batch = batch.to('cuda:0')
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'].to('cuda:0'), attention_mask=batch['attention_mask'].to('cuda:0'), labels=batch['labels'].to('cuda:0'))
            logits = outputs.logits

            batch = batch.to('cpu')
            logits = logits.to('cpu')

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_losses = loss.view(batch['input_ids'].size(0), -1)
            # now get the average loss just for the completion region (where mask == 1), in each row
            shift_mask = (batch['attention_mask'][..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
            masked_shift_losses = shift_losses * shift_mask
            # sum and divide by the number of 1s in the mask
            sum_loss = masked_shift_losses.sum(dim=1)

            n = sum_loss.size(0) // 4
            sum_loss_reshaped = sum_loss.view(n, 4).squeeze()

            # now we have a loss for each of the 4 completions
            # the one with the lowest loss should be the most likely

            pred = sum_loss_reshaped.argmin(dim=1)
            # _, pred = torch.topk(sum_loss_reshaped, n, largest=False)
            # pred = sum_loss.argmin().item()

            orgn_labels = batch['orgn_label'].view(n, 4).squeeze()
            orgn_labels = orgn_labels[:, 0]

            pred = pred.tolist()
            orgn_labels = orgn_labels.flatten().tolist()

            for p, l in zip(pred, orgn_labels):
                num_eval += 1
                if p == l:
                    num_correct += 1
            
        pbar.set_description(f"Evaluating Acc: step {step}/{len(acc_loader)} completed (acc: {num_correct / num_eval * 100:.4f})")

    pbar.close()
    
    print("acc :", num_correct / num_eval * 100)

    if wandb_run:
        wandb_run.log({
            'acc' : num_correct / num_eval * 100
        })
        