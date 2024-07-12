import copy
import torch
import datasets
# import functools


# for train
# ctx + 정답 ending => loss 낮추기 (same as samsum)

# for eval
# ctx + endings => endings 중 가장 loss가 낮은 ending을 output label로 선택해서 acc

def get_hella_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("Rowan/hellaswag", split=split)

    def render(sample):
        ctx = sample["ctx"]
        label = sample["label"]
        endings = sample["endings"]

        ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
        end_tokens = tokenizer.encode(endings[int(label)], add_special_tokens=False)
        tokens = ctx_tokens + end_tokens

        sample = {
            "input_ids": tokens,
            "attention_mask" : [1] * len(tokens),
            "labels": copy.deepcopy(tokens),
        }
        return sample
    
    dataset = dataset.map(render, remove_columns=list(dataset.features))

    return dataset

class HellaDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.dataset = datasets.load_dataset("Rowan/hellaswag", split="validation")
        self.tokenizer = tokenizer
        self.samples = []

        for i in range(0, len(self.dataset)):
            sample = self.dataset[i]
            ctx = sample["ctx"]
            label = int(sample["label"])
            endings = sample["endings"]

            for idx, end in enumerate(endings):
                ctx_tokens = self.tokenizer.encode(ctx, add_special_tokens=False)
                end_tokens = self.tokenizer.encode(end, add_special_tokens=False)
                tokens = ctx_tokens + end_tokens

                sample = {
                    "input_ids": tokens,
                    "attention_mask" : [1] * len(tokens),
                    "labels": [-100] * len(ctx_tokens) + end_tokens,
                    "orgn_label" : label,
                }

                self.samples.append(sample)

    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]