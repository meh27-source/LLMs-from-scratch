# Understanding BPE Tokenization with tiktoken

## Overview

This project explores how modern Large Language Models (LLMs) process text before it is fed into a neural network.

The focus is on Byte Pair Encoding (BPE), the subword tokenization algorithm used by GPT-2 and many other language models.

After implementing a tokenizer from scratch, this project investigates how OpenAI's production tokenizer works using the `tiktoken` library.

---

## Concepts Explored

### Byte Pair Encoding (BPE)

Byte Pair Encoding is a data-compression inspired tokenization technique.

Instead of storing every word as a separate token, BPE:

* Keeps common words as single tokens
* Splits rare words into meaningful subwords
* Reduces vocabulary size
* Handles previously unseen words effectively

Example:

```
highest
estimate
```

Both words contain the common subword:

```
est
```

BPE can learn this frequently occurring pattern and represent it as a reusable token.

---

## Why BPE is Important

Traditional word-level tokenizers suffer from the Out-of-Vocabulary (OOV) problem.

Example:

```
electromagnetism
```

If the word was never seen during training, a word-level tokenizer may fail.

BPE solves this by decomposing words into smaller subword units:

```
electro + magnet + ism
```

Benefits:

* Smaller vocabulary
* Better handling of rare words
* Retention of semantic information
* Efficient representation of text

---

## Technologies Used

* Python
* tiktoken
* GPT-2 Encoding

---

## Example

Input:

```text
Hello, do you like tea?<|endoftext|>
In the sunlit terraces of some unknown place.
```

The tokenizer converts the text into token IDs:

```python
token_ids = tokenizer.encode(text)
```

These IDs can then be decoded back into the original text.

---

## What I Learned

Through this project I gained a deeper understanding of:

* Tokenization in LLMs
* Byte Pair Encoding (BPE)
* Vocabulary construction
* Special tokens
* Text encoding and decoding
* How GPT models process language

---

## Future Improvements

* Visualize token boundaries
* Compare GPT-2 and GPT-4 tokenization
* Analyze token frequency distributions
* Build a custom BPE tokenizer from scratch

---

## Author

Mehak Batra
