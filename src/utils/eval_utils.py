


"""
code for final evaluation in inference settings 
"""

import os
from functools import partial 
from tqdm import tqdm
import evaluate
from utils.utils import jsave
import torch

# def final_eval_rouge(model, eval_dataloader, accelerator, curr_train_epoch):
#     metric = evaluate.load("rouge") ##TODO(HJ) need to change
    
#     with torch.no_grad():
#         outputs = model.generate(**batch)
#     predictions = outputs.logits.argmax(dim=-1)
#     # gather all inferences across chips
#     accelerator.wait_for_everyone()
#     ##(TODO)HJ multigpu gather inference 
#     # predictions = accelerator.pad_across_processes(
#     #         predictions, dim=1, pad_index=tokenizer.pad_token_id)

#     # references = accelerator.pad_across_processes(
#     #         batch['labels'], dim=1, pad_index=tokenizer.pad_token_id).detach().cpu().numpy()

    # references= torch.where(batch['labels'] != -100, batch['labels'], tokenizer.pad_token_id)
    # predictions= torch.where(batch['labels'] != -100, predictions, tokenizer.pad_token_id)

    # predictions = tokenizer.batch_decode(predictions.detach().cpu().numpy(), skip_special_tokens=True)
    # references = tokenizer.batch_decode(references.detach().cpu().numpy(), skip_special_tokens=True)
    
    # if step < train_config.debug_n_example:
    #     print("predictions", predictions)
    #     print("references", references)
    
    # predictions, references = accelerator.gather_for_metrics((predictions, references))

#     metric.add_batch(
#         predictions=predictions,
#         references=references,
#     )
#     eval_metric = metric.compute()
#     # Use accelerator.print to print only on the main process.
#     accelerator.print(f"epoch {curr_train_epoch}:", eval_metric)

def get_rouge_score(predictions, references):
    metric = evaluate.load("rouge")
    metric.add_batch(predictions=predictions, references=references)
    eval_metric = metric.compute()
    return eval_metric
    

GET_METRICS = {"rouge": get_rouge_score,}


def get_metrics(results, eval_config):
    calculator = GET_METRICS[eval_config.metric]
    
    inferences = [result['inference'] for result in results]
    references = [result['reference'] for result in results]

    metrics = calculator(inferences, references)

    print(f"{eval_config.model_name} | {eval_config.dtype} | {eval_config.dataset} | {eval_config.metric} results")
    print(metrics)

    if eval_config.save_metrics:
        output_path = os.path.join(eval_config.output_dir, "metrics.json")
        jsave(output_path, metrics)
        ## save metrics
    
def get_inference(model, eval_dataloader, tokenizer, eval_config, accelerator):
    
    model.eval()
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    
    
    tot_pairs = list()
   
    for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
        references = batch.pop("labels")
        #print(tokenizer.batch_decode(batch['input_ids'].detach().cpu().numpy(), skip_special_tokens=True))
        input_ids = batch['input_ids'].to(accelerator.device)
        
        
        with torch.no_grad():
            inferences = model.generate(
                                    input_ids,
                                    do_sample=eval_config.do_sample,
                                    temperature=eval_config.temperature, 
                                    max_new_tokens=eval_config.max_new_tokens)

        input_len = input_ids.size(1)
        inferences = tokenizer.batch_decode(inferences[:,input_len:].detach().cpu().numpy(), skip_special_tokens=True)
        references = tokenizer.batch_decode(references.detach().cpu().numpy(), skip_special_tokens=True)
        
        # print(inferences)
        # print(references)

        for i in range(input_ids.size(0)):
            idx = len(tot_pairs) + i
            tot_pairs.append(dict(index=idx, inference=inferences[i], reference=references[i]))
        
    if eval_config.save_inference:
        output_path = os.path.join(eval_config.output_dir, "inferences.json")
        jsave(output_path, tot_pairs)
    
    return tot_pairs