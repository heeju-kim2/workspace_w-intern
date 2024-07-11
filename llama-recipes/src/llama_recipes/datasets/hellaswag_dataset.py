import copy
import torch
import datasets
# import functools


# for train
# ctx + 정답 ending => loss 낮추기 (same as samsum)

# for eval
# ctx + endings => endings 중 가장 loss가 낮은 ending을 output label로 선택해서 acc

def render_for_acc(example, tokenizer):
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
        end_tokens = tokenizer.encode(end)
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

def render_for_train(example, tokenizer):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
    end_tokens = tokenizer.encode(endings[label], add_special_tokens=False)
    tokens = ctx_tokens + end_tokens

    data = {
        "input_ids": tokens,
        "attention_mask": [1] * len(tokens),
        "labels": copy.deepcopy(tokens),
    }

    return data

# class HellaDataset(torch.utils.data.Dataset):
#     def __init__(self, tokenizer, split):
#         self.dataset = datasets.load_dataset("Rowan/hellaswag", split=split)
#         self.tokenizer = tokenizer
#         self.split = split
    
#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         if self.split == 'acc':
#             data, tokens, mask, label = render_example(self.dataset[idx], self.tokenizer)
#             return data, tokens, mask, label
#         else:
#             data = render_for_train(self.dataset[idx], self.tokenizer)
#             return data

def get_hella_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("Rowan/hellaswag")

    return dataset

def get_preprocessed_hella(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("Rowan/hellaswag", split=split)

    def render(sample):
        ctx = sample["ctx"]
        label = sample["label"]
        endings = sample["endings"]

        ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
        end_tokens = tokenizer.encode(endings[label], add_special_tokens=False)
        tokens = ctx_tokens + end_tokens

        sample = {
            "input_ids": tokens,
            "attention_mask" : [1] * len(tokens),
            "labels": copy.deepcopy(tokens),
        }
        return sample
    
    # dataset = dataset.map(render, remove_columns=list(dataset.features))
    dataset = dataset.map(render)

    return dataset

def prepare_hella_data(tokenizer):
    dataset = datasets.load_dataset("Rowan/hellaswag", split="validation")

    def render(sample):
        ctx = sample["ctx"]
        label = sample["label"]
        endings = sample["endings"]
        samples = []
        for end in endings:
            ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
            end_tokens = tokenizer.encode(end, add_special_tokens=False)
            tokens = ctx_tokens + end_tokens

            sample = {
                "input_ids": tokens,
                "attention_mask" : [0] * len(ctx_tokens) + [1] * len(end_tokens),
                "labels": copy.deepcopy(tokens),
            }
            samples.append(sample)
        return samples

    dataset = dataset.map(render)

    return dataset