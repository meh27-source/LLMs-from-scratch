
# Simple LLM Tokenizer from Scratch

## Overview

This project implements a simple tokenizer from scratch in Python as part of my journey to understand the foundations of Large Language Models (LLMs).

The tokenizer performs:

* Text preprocessing using regular expressions
* Vocabulary creation
* Token-to-ID encoding
* ID-to-token decoding
* Handling of unknown words using `<|unk|>`
* Handling of document boundaries using `<|endoftext|>`

This project is educational and aims to demonstrate the basic principles behind tokenization before moving on to more advanced approaches such as Byte Pair Encoding (BPE).

---

## Features

### Tokenization

The input text is split into tokens using regular expressions.

### Vocabulary Construction

* Unique tokens are extracted from the training corpus.
* Tokens are sorted alphabetically.
* Each token is assigned a unique integer ID.

### Encoding

Converts tokens into their corresponding token IDs.

Example:

Text:

Hello world

Encoded:

[15, 42]

### Decoding

Converts token IDs back into text.

### Special Tokens

`<|unk|>`

Used for words that are not present in the vocabulary.

`<|endoftext|>`

Used to separate different documents or text sequences.

---

## Project Structure

tokenizer.py

Main implementation containing:

* Vocabulary creation
* Tokenizer V1
* Tokenizer V2
* Encoding and decoding examples

---

## Concepts Learned

Through this project I explored:

* Text preprocessing
* Regular expressions
* Vocabulary generation
* Dictionary-based token lookup
* Token IDs
* Encoding and decoding pipelines
* Unknown token handling
* Context tokens used in language models

---

## Future Improvements

* Byte Pair Encoding (BPE)
* Subword tokenization
* Better punctuation handling
* GPT-style tokenizer implementation
* Training a small language model using the generated tokens

---

## Technologies Used

* Python
* Regular Expressions (`re` module)

---

## Author

Mehak Batra

Built while learning the fundamentals of NLP and Large Language Models.