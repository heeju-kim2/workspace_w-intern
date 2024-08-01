import json
import os
import re
import copy
import torch


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

INVALID_ANS = "[invalid]"
PREAMBLE = """As an expert problem solver solve step by step the following mathematical question. 
At the end of steps, give the final answer in \" The answer is number.\" format."""
TEMPLATE = """Q: {question}\nA:"""
PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8."""

def read_jsonl(path: str):
    path = 'workspace_w-intern/src/dataset_srcs/gsm8k/' + path
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"].replace("####", "The answer is"))

    print(f"{len(examples)} {split} examples")
    return examples

def find_numbers(x: str):
  """Finds all numbers in a string."""
  # Search for number, possibly negative (hyphen), with thousand separators
  # (comma), and with a decimal point (period inbetween digits).
  numbers = re.compile(
      r'-?[\d,]*\.?\d+',
      re.MULTILINE | re.DOTALL | re.IGNORECASE,
  ).findall(x)
  return numbers


def find_number(x: str, answer_delimiter: str = 'The answer is') -> str:
  """Finds the most relevant number in a string."""
  # If model uses the answer delimiter, then select the first number following
  # that format.
  if answer_delimiter in x:
    answer = x.split(answer_delimiter)[-1]
    numbers = find_numbers(answer)
    if numbers:
      return numbers[0]

  # In general, select the last number in the string.
  numbers = find_numbers(x)
  if numbers:
    return remove_comma(numbers[-1])
  return ''

def remove_comma(x: str) -> str:
  # Example: 5,600 -> 5600
  return x.replace(',', '')

def is_correct(model_completion, gt_example):
    gt_answer = find_number(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return find_number(model_completion) == gt_answer

# Batch Econding
class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True, only_qns=False, few_shot=False, no_prompt=False):
        self.examples = examples
        if not no_prompt:
           self.qns = [tokenizer.bos_token + " " + B_INST + PREAMBLE + '\n' + (PROMPT if few_shot else "") + '\n' + TEMPLATE.format(question=ex["question"]) + E_INST for ex in self.examples]
        else:
           self.qns = [TEMPLATE.format(question=ex["question"]) for ex in self.examples]
        # self.qns = [TEMPLATE.format(question=ex["question"]) for ex in self.examples]
        self.ans = [ex["answer"] + tokenizer.eos_token for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.labels = [find_number(ex["answer"]) for ex in self.examples]
        self.loss_on_prefix = loss_on_prefix
        self.only_qns = only_qns

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]

        if not self.only_qns:
            mask = (
                ([int(self.loss_on_prefix)] * len(qn_tokens))
                + ([1] * len(ans_tokens))
            )
            input_ids = qn_tokens + ans_tokens
            labels = [-100] * len(qn_tokens) + ans_tokens
        else:
            mask = (
                [int(self.loss_on_prefix)] * len(qn_tokens)
            )

            input_ids = qn_tokens
            labels = qn_tokens

        return dict(input_ids=input_ids, attention_mask=mask, labels=labels)

def get_gsm8k_dataset(
    dataset_config, tokenizer, split
):
    few_shot = True if split in dataset_config.few_shot else False
    only_qns = True if split == "EM" else False

    if split == "train":
        examples = get_examples("train")
    elif split == "EM":
        examples = get_examples("test")
    else:
        examples = get_examples("test")

    print("split: ", split, ", few_shot: ", few_shot)
    dataset = GSMDataset(
        tokenizer=tokenizer,
        examples=examples,
        only_qns=only_qns,
        few_shot=few_shot,
        no_prompt=dataset_config.no_prompt
    )

    return dataset