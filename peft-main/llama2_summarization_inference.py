import argparse
import torch
import os
import pandas as pd
import evaluate
from datasets import load_dataset
import pickle
import warnings

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, LlamaForCausalLM
import tqdm

metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")


def prepare_instructions(dialogues, summaries):
    instructions = []

    prompt = (
        f"Summarize this dialog:\n{{dialogue}}\n---\nSummary:\n"
    )

    for dialogue, summary in zip(dialogues, summaries):
        example = prompt.format(
            dialogue=dialogue,
        )
        instructions.append(example)

    return instructions


def prepare_samsum_data():
    dataset = load_dataset("samsum")
    val_dataset = dataset["test"]

    dialogues = val_dataset["dialogue"]
    summaries = val_dataset["summary"]
    val_instructions = prepare_instructions(dialogues, summaries)

    return val_instructions, summaries


def main():
    val_instructions, summaries = prepare_samsum_data()

    # unpatch flash attention    unplace_flash_attn_with_attn()

    # load base LLM model and tokenizer
    model_name = "SalmanFaroz/Llama-2-7b-samsum"
    lr = 3e-2
    num_epochs = 1
    batch_size = 1

    # setup()

    model = LlamaForCausalLM.from_pretrained(model_name).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []
    for instruct, summary in tqdm.tqdm(zip(val_instructions, summaries), desc="evaluating"):
        input_ids = tokenizer(
            instruct, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=1e-2,
            )
            result = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0]
            result = result[len(instruct) :]
            results.append(result)
            # print(f"Instruction:{instruct}")
            # print(f"Summary:{summary}")
            # print(f"Generated:{result}")
            # print("----------------------------------------")

    # compute metric
    rouge = metric.compute(predictions=results, references=summaries, use_stemmer=True)

    metrics = {metric: round(rouge[metric] * 100, 2) for metric in rouge.keys()}

    print(metrics)


if __name__ == "__main__":
    main()
