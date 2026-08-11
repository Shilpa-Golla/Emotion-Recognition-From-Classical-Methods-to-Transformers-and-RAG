from collections import Counter

import nltk
import spacy

from datasets import load_dataset
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import config


# -----------------------------------
# DOWNLOAD WORDNET IF NEEDED
# -----------------------------------

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


# -----------------------------------
# INITIALIZE NLP TOOLS
# -----------------------------------

porter = PorterStemmer()

snowball = SnowballStemmer("english")

wordnet_lemmatizer = WordNetLemmatizer()

nlp = spacy.load("en_core_web_sm")


# -----------------------------------
# INDIVIDUAL METHODS
# -----------------------------------

def porter_stem(word):
    return porter.stem(word)


def snowball_stem(word):
    return snowball.stem(word)


def wordnet_lemma(word):
    return wordnet_lemmatizer.lemmatize(word)


def spacy_lemma(word):
    doc = nlp(word)

    if len(doc) > 0:
        return doc[0].lemma_

    return word


# -----------------------------------
# VOCABULARY FUNCTION
# -----------------------------------

def build_vocabulary(texts):
    vocabulary = set()

    for text in texts:
        words = text.lower().split()

        for word in words:
            vocabulary.add(word)

    return vocabulary


def transform_vocabulary(vocabulary, function):
    transformed = set()

    for word in vocabulary:
        transformed.add(function(word))

    return transformed


# -----------------------------------
# REDUCTION CALCULATION
# -----------------------------------

def calculate_reduction(original_size, new_size):

    reduction = (
        (original_size - new_size)
        / original_size
    ) * 100

    return reduction


# -----------------------------------
# MAIN STAGE
# -----------------------------------

def run(config_module=config):

    print("\nStarting Stemming & Lemmatization Stage...")

    dataset = load_dataset(
        config_module.DATASET_NAME
    )

    train_texts = dataset["train"][
        config_module.TEXT_COLUMN
    ]

    # Use a subset for speed
    sample_texts = train_texts[
        :config_module.STEM_LEMMA_SAMPLE_SIZE
    ]

    # -----------------------------------
    # EXAMPLE WORDS
    # -----------------------------------

    example_words = [
        "playing",
        "played",
        "studies",
        "running",
        "better",
        "cars",
        "happiness",
        "wolves"
    ]

    print("\n===== STEMMING VS LEMMATIZATION =====")

    print(
        f"{'Word':<15}"
        f"{'Porter':<15}"
        f"{'Snowball':<15}"
        f"{'WordNet':<15}"
        f"{'spaCy':<15}"
    )

    print("-" * 75)

    for word in example_words:

        print(
            f"{word:<15}"
            f"{porter_stem(word):<15}"
            f"{snowball_stem(word):<15}"
            f"{wordnet_lemma(word):<15}"
            f"{spacy_lemma(word):<15}"
        )

    # -----------------------------------
    # ORIGINAL VOCABULARY
    # -----------------------------------

    original_vocab = build_vocabulary(
        sample_texts
    )

    original_size = len(original_vocab)

    # -----------------------------------
    # PORTER
    # -----------------------------------

    porter_vocab = transform_vocabulary(
        original_vocab,
        porter_stem
    )

    # -----------------------------------
    # SNOWBALL
    # -----------------------------------

    snowball_vocab = transform_vocabulary(
        original_vocab,
        snowball_stem
    )

    # -----------------------------------
    # WORDNET
    # -----------------------------------

    wordnet_vocab = transform_vocabulary(
        original_vocab,
        wordnet_lemma
    )

    # -----------------------------------
    # SPACY
    # -----------------------------------

    spacy_vocab = transform_vocabulary(
        original_vocab,
        spacy_lemma
    )

    # -----------------------------------
    # RESULTS
    # -----------------------------------

    print("\n===== VOCABULARY REDUCTION =====")

    print(
        "Original vocabulary size:",
        original_size
    )

    methods = [
        ("Porter", len(porter_vocab)),
        ("Snowball", len(snowball_vocab)),
        ("WordNet", len(wordnet_vocab)),
        ("spaCy", len(spacy_vocab))
    ]

    for method_name, new_size in methods:

        reduction = calculate_reduction(
            original_size,
            new_size
        )

        print(
            f"{method_name:<10}"
            f"Vocabulary: {new_size:<8}"
            f"Reduction: {reduction:.2f}%"
        )

    print(
        "\nStemming & Lemmatization "
        "Stage Completed Successfully!"
    )