import copy
import datasets


def get_preprocessed_samsum(dataset_config, tokenizer, split, num_examples=None, do_eval=False):
    if num_examples:
        split += f"[:{num_examples}]"
    dataset = datasets.load_dataset("samsum", split=split)
    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    def apply_tokenizer(sample): #TODO(HJ) drop references for inference 
        return {
            "input_ids" : tokenizer(sample['prompt']).input_ids,
            "labels" : tokenizer(sample['summary']).input_ids,
        }
    
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    
    if do_eval:
        dataset = dataset.map(apply_tokenizer, remove_columns=list(dataset.features))
        return dataset

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

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch
    config=None
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    
    dataset = get_preprocessed_samsum(config, tokenizer, "validation", 2, True)

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=2, pin_memory=True)
    for batch in dataloader:
        print(batch)
        exit(0)    
    print(dataset[0])
