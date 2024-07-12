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
        activity = sample['activity_label']
        prompt = f"Now situation is {{activity}}.\n"

        # ctx_tokens = tokenizer.encode(tokenizer.bos_token + " " + ctx, add_special_tokens=False)
        ctx_tokens = tokenizer.encode(tokenizer.bos_token + " " + prompt.format(activity=activity) + ctx, add_special_tokens=False)
        end_tokens = tokenizer.encode(endings[int(label)] + tokenizer.eos_token, add_special_tokens=False)
        tokens = ctx_tokens + end_tokens

        sample = {
            "input_ids": tokens,
            "attention_mask" : [1] * len(tokens),
            "labels": [-100] * len(ctx_tokens) + end_tokens,
        }
        return sample
    
    dataset = dataset.map(render, remove_columns=list(dataset.features))

    return dataset

class HellaDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.dataset = datasets.load_dataset("Rowan/hellaswag", split="validation")
        self.tokenizer = tokenizer
        self.samples = []
        prompt = f"Now situation is {{activity}}.\n"

        for i in range(0, len(self.dataset)):
            sample = self.dataset[i]
            ctx = sample["ctx"]
            label = int(sample["label"])
            endings = sample["endings"]
            activity = sample['activity_label']

            for idx, end in enumerate(endings):
                ctx_tokens = self.tokenizer.encode(self.tokenizer.bos_token + " " + prompt.format(activity=activity) + ctx, add_special_tokens=False)
                # ctx_tokens = self.tokenizer.encode(self.tokenizer.bos_token + " " + ctx, add_special_tokens=False)
                end_tokens = self.tokenizer.encode(end + self.tokenizer.eos_token, add_special_tokens=False)
                tokens = ctx_tokens + end_tokens

                sample = {
                    "input_ids": tokens,
                    "attention_mask" : [1] * len(tokens),
                    "labels": [-100] * len(ctx_tokens) + end_tokens,
                    "orgn_label" : label,
                    "end_len" : len(end_tokens)
                }

                self.samples.append(sample)

    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]