import json
import os
import re
import copy
import torch


def read_jsonl(path: str):
    path = 'workspace_w-intern/llama-recipes/src/llama_recipes/datasets/gsm8k/' + path
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"])

    print(f"{len(examples)} {split} examples")
    return examples


INVALID_ANS = "[invalid]"
# PROMPT = f"Solve given question: {{question}} and Answer ### number"

def extract_answer(completion):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer

# Batch Econding
class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=False, only_qns=False):
        self.examples = examples
        self.qns = [tokenizer.bos_token + ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] + tokenizer.eos_token for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.labels = [extract_answer(ex["answer"]) for ex in self.examples]
        self.loss_on_prefix = loss_on_prefix
        self.only_qns = only_qns
        # self.max_len = max(
        #     [
        #         len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
        #         for i in range(len(self.examples))
        #     ]
        # )
        # print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        # pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        # tokens = qn_tokens + ans_tokens + pad_tokens
        if not self.only_qns:
            mask = (
                ([int(self.loss_on_prefix)] * len(qn_tokens))
                + ([1] * len(ans_tokens))
                # + ([0] * len(pad_tokens))
            )

            tokens = qn_tokens + ans_tokens
        else:
            mask = (
                [int(self.loss_on_prefix)] * len(qn_tokens)
            )

            tokens = qn_tokens

        # tokens = torch.tensor(tokens)
        # mask = torch.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask, labels=copy.deepcopy(tokens))


# Ecode one by one
# class GSMDataset(torch.utils.data.Dataset):
#     def __init__(self, tokenizer, examples, loss_on_prefix=True):
#         self.examples = examples
#         self.qns = [ex["question"] for ex in self.examples]
#         self.ans = [ex["answer"] for ex in self.examples]
#         if None in self.qns or None in self.ans:
#             print("NONE IS HERE~~")
#         # self.qns = tokenizer.encode(self.qns, padding=False)
#         # self.ans = tokenizer.encode(self.ans, padding=False)
#         self.loss_on_prefix = loss_on_prefix
#         self.tokenizer = tokenizer
#         # self.max_len = max(
#         #     [
#         #         len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
#         #         for i in range(len(self.examples))
#         #     ]
#         # )
#         # print(f"Max tokens: {self.max_len}")

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         qn_tokens = self.qns[idx]
#         ans_tokens = self.ans[idx]
#         print(qn_tokens)
#         print(ans_tokens)
#         qn_tokens = self.tokenizer.encode(qn_tokens, padding=False)
#         ans_tokens = self.tokenizer.encode(ans_tokens, padding=False)
#         # pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
#         # tokens = qn_tokens + ans_tokens + pad_tokens
#         mask = (
#             ([int(self.loss_on_prefix)] * len(qn_tokens))
#             + ([1] * len(ans_tokens))
#             # + ([0] * len(pad_tokens))
#         )

#         # tokens = torch.tensor(tokens)
#         # mask = torch.tensor(mask)
#         return dict(input_ids=qn_tokens, attention_mask=mask, labels=ans_tokens)