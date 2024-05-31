from tqdm import tqdm
from datasets import load_dataset

# save to huggingface using api
from huggingface_hub import HfApi

def truncate_text(examples, max_length=200):
    splitted = examples["text"].split()
    return " ".join(splitted[:max_length])

api = HfApi()

# create a new repo
api.create_repo("mini_wiki", repo_type="dataset", exist_ok=True)

# split and get first 500 words
ds = load_dataset("wikimedia/wikipedia", "20231101.en")["train"]

# sample 1000 examples from the dataset
ds = ds.shuffle(seed=42).select(range(100_000))
ds_trunc = ds.map(lambda x: {"text": truncate_text(x)}, remove_columns=["text"])

# take only the first 500 words from each example

n_to_str = {
    100: "100",
    1000: "1k",
    5000: "5k",
    10_000: "10k",
    50_000: "50k",
    100_000: "100k",
}

for n in n_to_str:
    ds_sampled = ds_trunc.select(range(n))
    ds_sampled_full = ds.select(range(n))
    
    config = n_to_str[n]
    ds_sampled.push_to_hub(
        repo_id="xhluca/mini_wiki",
        config_name=config,
        commit_message=f"Add {config} examples",
        split="partial",
    )

    ds_sampled_full.push_to_hub(
        repo_id="xhluca/mini_wiki",
        config_name=config,
        commit_message=f"Add {config} examples",
        split="full",
    )