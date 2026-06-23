import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


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
            target_chunk = token_ids[
                i + 1:i + max_length + 1
            ]

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


with open(
    "the-verdict.txt",
    "r",
    encoding="utf-8"
) as f:
    raw_text = f.read()


print("PyTorch version:", torch.__version__)

max_length = 4

dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)

data_iter = iter(dataloader)

inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInput Shape:", inputs.shape)

vocab_size = 50257
embedding_dim = 256

token_embedding_layer = torch.nn.Embedding(
    vocab_size,
    embedding_dim
)

token_embeddings = token_embedding_layer(inputs)

print(
    "\nToken Embedding Shape:",
    token_embeddings.shape
)

position_embedding_layer = torch.nn.Embedding(
    max_length,
    embedding_dim
)

position_embeddings = position_embedding_layer(
    torch.arange(max_length)
)

print(
    "Position Embedding Shape:",
    position_embeddings.shape
)

input_embeddings = (
    token_embeddings +
    position_embeddings
)

print(
    "Input Embedding Shape:",
    input_embeddings.shape
)

print(
    "\nFirst Input Embedding Vector:"
)

print(
    input_embeddings[0][0]
)

# TOKEN EMBEDDINGS

# GPT first converts text into token IDs using Byte Pair Encoding (BPE).
# Token IDs are only labels, so the model cannot learn meaningful patterns from them directly.
# Each token ID is therefore converted into a vector called a token embedding.
# These vectors allow the model to represent words mathematically.
# Similar words often end up with similar vectors because the embeddings are learned during training.
# The embedding layer can be thought of as a lookup table that returns a vector for a given token ID.
# At the beginning of training, the embedding values are random and gradually improve as the model learns.
# The embedding matrix depends on two things: the vocabulary size and the embedding dimension.

# POSITIONAL EMBEDDINGS

# Token embeddings contain meaning, but they do not tell the model where a word appears in a sentence.
# The same word receives the same token embedding whether it appears at the beginning or the end of a sentence.
# To solve this problem, transformers use positional embeddings.
# Positional embeddings help the model understand the order of words in a sequence.
# Absolute positional embeddings focus on exact positions, while relative positional embeddings focus on distances between tokens.
# Positional embeddings have the same dimension as token embeddings so that they can be added together.

# INPUT EMBEDDINGS

# The final input representation is created by adding token embeddings and positional embeddings.
# This gives each token both semantic information and positional information.
# These combined vectors are called input embeddings.
# Input embeddings are the actual inputs that are passed to the transformer model.

# PROGRESS

# Raw text is converted into token IDs using the tokenizer.
# Token IDs are converted into token embeddings.
# Positional embeddings are added to provide information about word order.
# The resulting input embeddings are passed to the transformer for further processing.