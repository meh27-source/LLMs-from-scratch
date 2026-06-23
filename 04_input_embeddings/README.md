# Input Embeddings

## Why I Added This Section

In the previous stage, I created a DataLoader that converts raw text into batches of token IDs that can be processed efficiently during training.

While token IDs are useful for identifying words and subwords, they are still just numerical labels. A language model cannot learn meaningful relationships directly from token IDs.

This raised an important question:

**How does a language model transform token IDs into a form that a neural network can actually understand?**

This section explores the answer through token embeddings and positional embeddings.

---

## Building on the Previous Stage

At the end of the DataLoader stage, the model receives batches of token IDs such as:

```python
[40, 367, 2885, 1464]
```

These values represent tokens in the vocabulary, but they do not contain any information about meaning or context.

Before any learning can happen, each token ID must be converted into a vector representation.

---

## Token Embeddings

A token embedding is a dense vector associated with a token ID.

Instead of representing a token with a single number, the model represents it using hundreds of numerical features.

These vectors are stored in an embedding matrix.

The embedding layer acts like a lookup table:

```text
Token ID → Embedding Vector
```

At the beginning of training, the embedding values are random. As training progresses, the vectors gradually learn meaningful relationships between words and phrases.

This allows the model to capture semantic information rather than working with simple token IDs.

---

## Positional Embeddings

Token embeddings contain information about meaning, but they do not contain information about word order.

For example, the word "apple" receives the same token embedding regardless of where it appears in a sentence.

Transformers solve this problem using positional embeddings.

Positional embeddings provide information about where a token appears within a sequence so that the model can distinguish between different word orders.

---

## Creating Input Embeddings

The final representation is obtained by combining:

* Token Embeddings
* Positional Embeddings

The resulting vectors are called **Input Embeddings**.

These embeddings contain both semantic information and positional information and serve as the actual input to the transformer architecture.

---

## What This Program Does

This implementation:

* Loads text from *The Verdict*
* Creates batches of token IDs using the previously developed DataLoader
* Converts token IDs into token embeddings
* Creates positional embeddings
* Combines both embeddings into final input embeddings
* Displays the resulting tensor shapes

---

## Technologies Used

* Python
* PyTorch
* tiktoken

---

## Concepts Learned

During this stage I explored:

* Byte Pair Encoding (BPE)
* Token IDs
* Embedding Layers
* Vocabulary Size
* Embedding Dimensions
* Token Embeddings
* Positional Embeddings
* Input Embeddings

---

## What I Learned

This stage helped me understand that language models do not process words directly. Instead, they operate on high-dimensional vector representations that are learned during training.

I also learned how semantic information and positional information are combined before any attention mechanism is applied.

Most importantly, this section connected ideas from natural language processing, machine learning and linear algebra, showing how language can be represented mathematically inside a transformer model.

---

## Next Step (UPDATED LATER)

Now that token IDs have been converted into input embeddings, the next challenge is understanding how tokens interact with one another.

In the next stage, I will implement self-attention and explore how transformers determine which words in a sequence are most relevant to each other.
