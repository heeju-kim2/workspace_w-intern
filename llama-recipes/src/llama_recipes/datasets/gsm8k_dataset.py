from llama_recipes.datasets.gsm8k.dataset import get_examples, GSMDataset

# custom_dataset(dataset='custom_dataset', file='examples/custom_dataset.py', train_split='train', test_split='validation')
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

    dataset = GSMDataset(
        tokenizer=tokenizer,
        examples=examples,
        only_qns=only_qns,
        few_shot=few_shot
    )

    return dataset