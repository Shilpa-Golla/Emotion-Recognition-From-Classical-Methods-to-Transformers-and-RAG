import os
import re
from collections import Counter

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

import config
from datasets import load_dataset


# -----------------------------------
# 1. WHITESPACE TOKENIZER
# -----------------------------------

def whitespace_tokenize(text):
    return text.split()


# -----------------------------------
# 2. PUNCTUATION-AWARE TOKENIZER
# -----------------------------------

def punctuation_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


# -----------------------------------
# 3. CHARACTER TOKENIZER
# -----------------------------------

def character_tokenize(text):
    return list(text)


# -----------------------------------
# 4. BPE TOKENIZER
# -----------------------------------

def train_bpe_tokenizer(train_texts, config_module=config):

    tokenizer = Tokenizer(
        BPE(unk_token="[UNK]")
    )

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=config_module.BPE_VOCAB_SIZE,
        min_frequency=config_module.BPE_MIN_FREQUENCY,
        special_tokens=[
            "[UNK]",
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
        ]
    )

    tokenizer.train_from_iterator(
        train_texts,
        trainer=trainer
    )

    return tokenizer


# -----------------------------------
# CREATE TRAIN VOCABULARY
# -----------------------------------

def build_vocabulary(texts, tokenizer_function):

    vocabulary = set()

    for text in texts:
        tokens = tokenizer_function(text.lower())

        for token in tokens:
            vocabulary.add(token)

    return vocabulary


# -----------------------------------
# OOV CALCULATION
# -----------------------------------

def calculate_oov_rate(texts, tokenizer_function, train_vocabulary):

    total_tokens = 0
    oov_tokens = 0

    for text in texts:

        tokens = tokenizer_function(text.lower())

        for token in tokens:

            total_tokens += 1

            if token not in train_vocabulary:
                oov_tokens += 1

    if total_tokens == 0:
        return 0.0

    oov_rate = (oov_tokens / total_tokens) * 100

    return oov_rate


# -----------------------------------
# MAIN TOKENIZATION STAGE
# -----------------------------------

def run(config_module=config):

    print("\nStarting Tokenization Stage...")

    dataset = load_dataset(
        config_module.DATASET_NAME
    )

    train_texts = dataset["train"][
        config_module.TEXT_COLUMN
    ]

    test_texts = dataset["test"][
        config_module.TEXT_COLUMN
    ]

    example = train_texts[0]

    print("\n===== ORIGINAL SENTENCE =====")
    print(example)

    # -----------------------------------
    # Whitespace
    # -----------------------------------

    whitespace_tokens = whitespace_tokenize(example)

    print("\n===== WHITESPACE TOKENIZER =====")
    print(whitespace_tokens)
    print(
        "Token count:",
        len(whitespace_tokens)
    )

    # -----------------------------------
    # Punctuation-aware
    # -----------------------------------

    punctuation_tokens = punctuation_tokenize(example)

    print("\n===== PUNCTUATION-AWARE TOKENIZER =====")
    print(punctuation_tokens)
    print(
        "Token count:",
        len(punctuation_tokens)
    )

    # -----------------------------------
    # Character
    # -----------------------------------

    character_tokens = character_tokenize(example)

    print("\n===== CHARACTER TOKENIZER =====")
    print(character_tokens)
    print(
        "Token count:",
        len(character_tokens)
    )

    # -----------------------------------
    # Build vocabularies
    # -----------------------------------

    whitespace_vocab = build_vocabulary(
        train_texts,
        whitespace_tokenize
    )

    punctuation_vocab = build_vocabulary(
        train_texts,
        punctuation_tokenize
    )

    character_vocab = build_vocabulary(
        train_texts,
        character_tokenize
    )

    # -----------------------------------
    # OOV rates
    # -----------------------------------

    whitespace_oov = calculate_oov_rate(
        test_texts,
        whitespace_tokenize,
        whitespace_vocab
    )

    punctuation_oov = calculate_oov_rate(
        test_texts,
        punctuation_tokenize,
        punctuation_vocab
    )

    character_oov = calculate_oov_rate(
        test_texts,
        character_tokenize,
        character_vocab
    )

    # -----------------------------------
    # BPE
    # -----------------------------------

    print("\nTraining BPE tokenizer...")

    bpe_tokenizer = train_bpe_tokenizer(
        train_texts,
        config_module
    )

    bpe_encoding = bpe_tokenizer.encode(
        example
    )

    print("\n===== BPE TOKENIZER =====")
    print(bpe_encoding.tokens)
    print(
        "Token count:",
        len(bpe_encoding.tokens)
    )

    # Calculate BPE OOV
    total_bpe_tokens = 0
    unknown_bpe_tokens = 0

    for text in test_texts:

        encoded = bpe_tokenizer.encode(text)

        total_bpe_tokens += len(encoded.tokens)

        unknown_bpe_tokens += encoded.tokens.count(
            "[UNK]"
        )

    if total_bpe_tokens > 0:
        bpe_oov = (
            unknown_bpe_tokens /
            total_bpe_tokens
        ) * 100
    else:
        bpe_oov = 0.0

    # -----------------------------------
    # RESULTS
    # -----------------------------------

    print("\n===== OOV RATE COMPARISON =====")

    print(
        f"Whitespace OOV: "
        f"{whitespace_oov:.2f}%"
    )

    print(
        f"Punctuation OOV: "
        f"{punctuation_oov:.2f}%"
    )

    print(
        f"Character OOV: "
        f"{character_oov:.2f}%"
    )

    print(
        f"BPE OOV: "
        f"{bpe_oov:.2f}%"
    )

    # -----------------------------------
    # SAVE BPE TOKENIZER
    # -----------------------------------

    os.makedirs(
        config_module.OUTPUT_DIR,
        exist_ok=True
    )

    tokenizer_path = os.path.join(
        config_module.OUTPUT_DIR,
        "bpe_tokenizer.json"
    )

    bpe_tokenizer.save(
        tokenizer_path
    )

    print(
        "\nSaved BPE tokenizer:",
        tokenizer_path
    )

    print(
        "\nTokenization Stage "
        "Completed Successfully!"
    )