from llama_recipes.datasets.gsm8k.dataset import get_examples, GSMDataset

# custom_dataset(dataset='custom_dataset', file='examples/custom_dataset.py', train_split='train', test_split='validation')
def get_gsm8k_dataset(
    dataset_config, tokenizer, split
):

    if split == "train":
        examples = get_examples("train")
    else:
        examples = get_examples("test")

    dataset = GSMDataset(
        tokenizer=tokenizer,
        examples=examples,
    )

    return dataset

