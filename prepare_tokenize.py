import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import copy
import json
import argparse
import transformers
from typing import Dict, List
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_INDEX = LabelSmoother.ignore_index

def preprocess(
    list_data_dict: List,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    input_ids = tokenizer.apply_chat_template(list_data_dict, max_length=tokenizer.model_max_length, truncation=True)
    source_ids = tokenizer.apply_chat_template([l[:-1] for l in list_data_dict], add_generation_prompt=True, max_length=tokenizer.model_max_length, truncation=True)
    labels = copy.deepcopy(input_ids)
    source_lens = [len(source_id) for source_id in source_ids]
    for label, source_len in zip(labels, source_lens):
        label[:source_len] = [IGNORE_INDEX] * source_len
    return dict(input_ids=input_ids, labels=labels)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, help="The model name or path to the model.")
parser.add_argument("--input_path", type=str, help="Input Path")
parser.add_argument("--output_path", type=str, help="Output Path")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
)

with open(args.input_path, "r") as f:
    list_data_dict = json.load(f)

import time
start = time.time()
data_dict = preprocess(list_data_dict, tokenizer)
end = time.time()
print(f"Time taken: {end - start} seconds")

with open(args.output_path, "w") as f:
    json.dump(data_dict, f, indent=4)
