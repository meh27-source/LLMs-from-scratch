"""
tiktoken_demo.py

A simple demonstration of how GPT-2 tokenization works using OpenAI's
tiktoken library.

This project explores:
- Byte Pair Encoding (BPE)
- Tokenization and detokenization
- Special tokens
- GPT-2 vocabulary
"""

import tiktoken

tokenizer=tiktoken.get_encoding("gpt2")

text="Hello ,do you like tea?<|endoftext|> In the sunlit terraces of some unknown place."

ids=tokenizer.encode(text,allowed_special={"<|endoftext|>"})

print(ids)

strings=tokenizer.decode(ids)

print(strings)