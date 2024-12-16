import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import copy
import json
import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_INDEX = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field()

@dataclass
class DataArguments:
    data_path: str = field()
    num_proc: int = field()

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field()

def preprocess(
    list_data_dict: List,
    tokenizer: transformers.PreTrainedTokenizer,
    num_proc: int,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [tokenizer.apply_chat_template(l, tokenize=False) for l in list_data_dict]
    sources = [tokenizer.apply_chat_template(l[:-1], tokenize=False, add_generation_prompt=True) for l in list_data_dict]

    """Tokenize a list of strings."""
    def tokenize_text(text):
        return tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        examples_tokenized = list(executor.map(tokenize_text, examples))
        sources_tokenized = list(executor.map(tokenize_text, sources))

    input_ids = [tokenized.input_ids[0].tolist() for tokenized in examples_tokenized]
    labels = copy.deepcopy(input_ids)
    source_lens = [len(tokenized.input_ids[0]) for tokenized in sources_tokenized]
    for label, source_len in zip(labels, source_lens):
        label[:source_len] = [IGNORE_INDEX] * source_len
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, num_proc: int):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r") as json_file:
            list_data_dict = json.load(json_file)
        data_dict = preprocess(list_data_dict, tokenizer, num_proc=num_proc)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, List[int]]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, num_proc=data_args.num_proc)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False
    model = torch.compile(model)
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
