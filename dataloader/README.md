# GPT Dataset Preparation

## Project Overview

Before training a language model, raw text must be transformed into a format that a neural network can learn from.

This stage focuses on building the dataset pipeline used for GPT-style next-token prediction. The goal is not simply to collect text, but to understand how language data is prepared before it enters a Large Language Model.

The project explores the complete process from text preprocessing to the creation of training samples that can be used by a neural network.

---

## Why This Project?

When using modern AI systems such as ChatGPT, it is easy to focus only on the model architecture and overlook the importance of the data itself.

However, every language model learns from examples contained in its training dataset.

I wanted to understand how raw text is transformed into meaningful training examples and what happens before neural network training begins.

This project was created as part of my broader effort to build a GPT-style language model from scratch and understand every stage of the pipeline rather than treating it as a black box.

---

## Objectives

The main objectives of this project are:

* Understand GPT-style training data
* Explore text preprocessing techniques
* Work with tokenized text
* Create input-target training pairs
* Prepare data for neural network training
* Develop a deeper understanding of NLP data pipelines

---

## Dataset Pipeline

The workflow follows several steps:

### 1. Text Collection

Raw text is gathered from source documents and loaded into Python for processing.

### 2. Text Cleaning

Unnecessary formatting and inconsistencies are removed to create a cleaner training corpus.

### 3. Tokenization

The text is converted into tokens.

Modern GPT models commonly use Byte Pair Encoding (BPE), which allows efficient handling of both common words and rare subwords.

Example:

```text
Artificial intelligence is transforming technology.
```

↓

```text
[4512, 982, 318, 7411, 1250]
```

The model does not work directly with words. It works with token IDs.

### 4. Creating Training Samples

The tokenized text is transformed into input-target pairs.

Example:

Input:

```text
[4512, 982, 318]
```

Target:

```text
7411
```

The model learns to predict the next token based on previous tokens.

---

## Technologies Used

* Python
* NumPy
* Regular Expressions (Regex)
* Tokenization Techniques
* GPT-style Dataset Preparation

---

## Key Concepts Learned

Through this project I explored:

* Natural Language Processing fundamentals
* Tokenization and token IDs
* Byte Pair Encoding (BPE)
* Context windows
* Next-token prediction
* Dataset preparation for LLMs
* Input-target sequence generation

One important insight was that building a language model starts long before neural networks are introduced. A large portion of the work involves organizing data into a form that allows meaningful learning to occur.

---

## Example Workflow

```text
Raw Text
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
Token IDs
    ↓
Input-Target Pairs
    ↓
Training Dataset
```

---

## What I Learned

This project helped me understand:

* How GPT models receive training data
* Why tokenization is necessary
* How next-token prediction datasets are constructed
* The relationship between context and prediction
* Why data quality directly impacts model performance

Most importantly, I learned that effective machine learning begins with effective data preparation.

---

## Next Steps

The next stage of the project focuses on transforming token IDs into trainable vector representations using embedding layers.

These embeddings allow the model to capture semantic relationships between tokens and serve as the foundation for neural network training.
