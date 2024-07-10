import copy
import torch
import datasets
# import functools

def render_example(example, tokenizer):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = tokenizer.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = tokenizer.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


class HellaDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split):
        self.dataset = datasets.load_dataset("Rowan/hellaswag", split=split)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, tokens, mask, label = render_example(self.dataset[idx], self.tokenizer)

def get_hella_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("Rowan/hellaswag", split=split)

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }

        return sample
    
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

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


def prepare_samsum_data(tokenizer):
    dataset = datasets.load_dataset("samsum")
    val_dataset = dataset["test"]

    dialogues = val_dataset["dialogue"]
    summaries = val_dataset["summary"]

    instructions = prepare_instructions(dialogues, summaries)

    val_dataset = val_dataset.map(apply_prompt_template, remove_columns=list(val_dataset.features))

    def tokenize_add_label(instruction):
        prompt = tokenizer.encode(tokenizer.bos_token + instruction["prompt"], add_special_tokens=False)

        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * len(prompt),
            "labels" : [-100] * len(prompt)
        }

        return sample
    
    val_dataset = val_dataset.map(tokenize_add_label, remove_columns=list(val_dataset.features))

    return val_dataset, instructions, summaries