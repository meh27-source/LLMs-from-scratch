import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)

print("Total number of tokens:", len(enc_text))

enc_sample = enc_text[50:]

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print(f"Input  : {x}")
print(f"Target : {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    target = enc_sample[i]

    print("Context:", tokenizer.decode(context))
    print("Target :", tokenizer.decode([target]))


class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):

        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(
            txt,
            allowed_special={"<|endoftext|>"}
        )

        for i in range(
            0,
            len(token_ids) - max_length,
            stride
        ):

            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(
                torch.tensor(input_chunk)
            )

            self.target_ids.append(
                torch.tensor(target_chunk)
            )

    def __len__(self):

        return len(self.input_ids)

    def __getitem__(self, idx):

        return (
            self.input_ids[idx],
            self.target_ids[idx]
        )


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(
        txt,
        tokenizer,
        max_length,
        stride
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


print("\nPyTorch version:", torch.__version__)

dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=4,
    stride=1,
    shuffle=False
)

data_iter = iter(dataloader)

first_batch = next(data_iter)

print("\nFirst Batch:")
print(first_batch)


# NOTES

# GPT learns through next-token prediction.

# Example:
# Input  : [101, 102, 103, 104]
# Target : [102, 103, 104, 105]

# The target sequence is the input sequence shifted
# one position to the left.

# Context Length:
# Number of previous tokens visible to the model.

# Autoregressive Model:
# Each generated token becomes part of the input
# for the next prediction step.

# Self-Supervised Learning:
# No manual labels are required.
# Labels are automatically generated from the text itself.

# Raw Text
#    ↓
# Token IDs
#    ↓
# Input-Target Pairs
#    ↓
# Dataset
#    ↓
# DataLoader
#    ↓
# GPT Training

# Sliding Window:
# Creates overlapping training examples from
# a continuous token stream.
# Stride:
# Number of positions the window moves forward.
# Dataset:
# Stores all input-target pairs.
# DataLoader:
# Groups samples into batches and feeds them
# efficiently to the model.
# Batch Size:Number of sequences processed simultaneously.
# Smaller Batch Size :More frequent and noisier updates.
# Larger Batch Size:More stable updates but higher memory usage.
# Parallel Processing:During training, all tokens in a sequence are processed simultaneously.