import itertools

import torch
import datasets
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

def load_data(model_name: str,
              dataset_name: str,
              cache_dir: str,
              seq_len: int,
              batch_size: int,
              pin_memory: bool,
              shuffle: bool = False,
              ) -> itertools.cycle:
    # Load tokenizer
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    def prepare_alpaca(sample_raw):
        template = {
            "description": "A shorter template to experiment with.",
            "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }
        if len(sample_raw["input"]):
            sample_text = template["prompt_input"].format(
                instruction=sample_raw["instruction"], input=sample_raw["input"]
            )
        else:
            sample_text = template["prompt_no_input"].format(
                instruction=sample_raw["instruction"]
            )
        if len(sample_raw["output"]):
            sample_text += sample_raw["output"]
        # NOTE: no padding, as batch size = 1 for chunked fine-tuning co-serving
        # sample_tokens = tokenizer(sample_text, truncation=True, padding="max_length", max_length=seq_len)
        sample_tokens = tokenizer(sample_text, truncation=True, max_length=seq_len)
        return sample_tokens

    def prepare_flan_flat(sample_raw):
        if len(sample_raw["source"]):
            sample_text = sample_raw["source"]
        if len(sample_raw["target"]):
            sample_text += sample_raw["target"]
        # NOTE: no padding, as batch size = 1 for chunked fine-tuning co-serving
        # sample_tokens = tokenizer(sample_text, truncation=True, padding="max_length", max_length=seq_len)
        sample_tokens = tokenizer(sample_text, truncation=True, max_length=seq_len)
        return sample_tokens

    dataset = datasets.load_dataset(dataset_name, cache_dir=cache_dir)
    if dataset_name.find("alpaca") != -1:
        dataset = dataset.map(lambda sample_raw: prepare_alpaca(sample_raw), remove_columns=dataset["train"].column_names)
    elif dataset_name.find("flan") != -1:
        dataset['train'] = dataset['train'].select(range(100000)) # FIXME: only keep first 100000 samples for test efficiency
        dataset = dataset.map(lambda sample_raw: prepare_flan_flat(sample_raw), remove_columns=dataset["train"].column_names)
    else:
        raise NotImplemented()
    
    # FIXME: only keep long-enough samples (# tokens == seq_len) for throughput testing
    print(f"# long-enough samples: {sum([ 1 for input_ids in dataset['train']['input_ids'] if len(input_ids) == seq_len ])}")
    dataset['train'] = dataset['train'].filter(lambda tokens: len(tokens['input_ids']) == seq_len)

    dataloader = torch.utils.data.DataLoader(
        # NOTE: set shuffle = False by default for reproduction
        dataset["train"], shuffle=shuffle, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        batch_size=batch_size, pin_memory=pin_memory,
    )
    dataloader_iter = itertools.cycle(iter(enumerate(dataloader)))
    
    return dataloader_iter
