import re

with open("LLMs/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("total number of characters are ", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r"[-(:;/,?!@()]|--|\s", raw_text)
cleaned = []

for item in preprocessed:
    if item.strip():
        cleaned.append(item.strip())

print(len(cleaned))
print(cleaned[:30])

all_words = sorted(set(cleaned))
print(len(all_words))

vocab = {token: integer for integer, token in enumerate(all_words)}

print(vocab)


class simpletokeniser:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str_ = {i: s for s, i in vocab.items()}

    def encoder(self, text):
        preprocess = re.split(r"[-(:;/,?!@()]|--|\s", text)
        preprocess = [item.strip() for item in preprocess if item.strip()]
        ids = [self.str_to_int[s] for s in preprocess]
        return ids

    def decoder(self, ids):
        tokens = [self.int_to_str_[i] for i in ids]
        final_text = " ".join(tokens)
        final_text = re.sub(
            r"\s+([-(:;/,?!@()]|--)",
            r"\1",
            final_text
        )
        return final_text


tokeniser = simpletokeniser(vocab)

text = '''"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.'''

ids = tokeniser.encoder(text)

print(ids)
print(tokeniser.decoder(ids))

all_tokens = sorted(set(cleaned))
all_tokens.extend(["<|unk|>", "<|endoftext|>"])

vocab2 = {token: integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))

for item in list(vocab.items())[-5:]:
    print(item)


class simpletokeniser2:
    def __init__(self, vocab2):
        self.str_to_int = vocab2
        self.int_to_str_ = {i: s for s, i in vocab2.items()}

    def encoder(self, text):
        preprocess = re.split(r"[-(:;/,?!@()]|--|\s", text)

        preprocess = [
            item.strip()
            for item in preprocess
            if item.strip()
        ]

        preprocess = [
            item if item in self.str_to_int
            else "<|unk|>"
            for item in preprocess
        ]

        ids = [self.str_to_int[s] for s in preprocess]
        return ids

    def decoder(self, ids):
        tokens = [self.int_to_str_[i] for i in ids]

        final_text = " ".join(tokens)

        final_text = re.sub(
            r"\s+([-(:;/,?!@()]|--)",
            r"\1",
            final_text
        )

        return final_text


tokeniser2 = simpletokeniser2(vocab2)

text1 = "Hello,do you like tea?"
text2 = "his hands in the pockets of his velveteen coat"

text = "<|endoftext|> ".join((text1, text2))

print(text)

ids = tokeniser2.encoder(text)

print(ids)
print(tokeniser2.decoder(ids))


"""
1. Data Preparation
- Read text file
- Split text into tokens
- Remove whitespace

2. Vocabulary Creation
- Extract unique tokens
- Sort alphabetically
- Assign token IDs

3. Tokenizer V1
- Encoder: token -> ID
- Decoder: ID -> token

4. Special Tokens
- <|unk|> for unknown words
- <|endoftext|> for document boundaries

5. Tokenizer V2
- Handles unknown tokens
- Supports special context tokens

6. Future Improvements
- Byte Pair Encoding (BPE)
- SentencePiece
- Better punctuation handling
"""