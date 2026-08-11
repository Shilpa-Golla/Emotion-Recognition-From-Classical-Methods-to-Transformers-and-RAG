import re
import unicodedata
import string

import contractions
import emoji
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

import config


# Download NLTK stopwords if they are not already available
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text, config_module=config):
    """
    Clean one text according to the switches in config.py.
    """

    # 1. Unicode normalization
    if config_module.NORMALIZE_UNICODE:
        text = unicodedata.normalize("NFKC", text)

    # 2. HTML removal
    if config_module.REMOVE_HTML:
        text = BeautifulSoup(text, "html.parser").get_text(" ")

    # 3. URL removal
    if config_module.REMOVE_URLS:
        text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 4. Email removal
    if config_module.REMOVE_EMAILS:
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            " ",
            text,
        )

    # 5. Contraction expansion
    if config_module.EXPAND_CONTRACTIONS:
        text = contractions.fix(text)

    # 6. Emoji handling
    if config_module.HANDLE_EMOJIS:
        text = emoji.demojize(text, delimiters=(" ", " "))

    # 7. Lowercase
    if config_module.LOWERCASE:
        text = text.lower()

    # 8. Number removal
    if config_module.REMOVE_NUMBERS:
        text = re.sub(r"\d+", " ", text)

    # 9. Punctuation removal
    if config_module.REMOVE_PUNCTUATION:
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )

    # 10. Stopword removal
    if config_module.REMOVE_STOPWORDS:
        words = text.split()
        words = [
            word for word in words
            if word not in STOP_WORDS
        ]
        text = " ".join(words)

    # 11. Whitespace normalization
    if config_module.NORMALIZE_WHITESPACE:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def run(config_module=config):
    print("\nStarting Preprocessing Stage...")

    # Artificial example deliberately contains several things
    # our preprocessing pipeline is designed to handle.
    example = (
        "OMG!!! I can't believe this . "
        "Visit https://example.com or email test@example.com. "
        "<b>I LOVE NLP!</b>"
    )

    cleaned = preprocess_text(example, config_module)

    print("\n===== BEFORE PREPROCESSING =====")
    print(example)

    print("\n===== AFTER PREPROCESSING =====")
    print(cleaned)

    print("\n===== ACTIVE SETTINGS =====")
    print("Lowercase:", config_module.LOWERCASE)
    print("Remove punctuation:", config_module.REMOVE_PUNCTUATION)
    print("Remove numbers:", config_module.REMOVE_NUMBERS)
    print("Remove URLs:", config_module.REMOVE_URLS)
    print("Remove emails:", config_module.REMOVE_EMAILS)
    print("Remove HTML:", config_module.REMOVE_HTML)
    print("Normalize whitespace:", config_module.NORMALIZE_WHITESPACE)
    print("Normalize Unicode:", config_module.NORMALIZE_UNICODE)
    print("Expand contractions:", config_module.EXPAND_CONTRACTIONS)
    print("Handle emojis:", config_module.HANDLE_EMOJIS)
    print("Remove stopwords:", config_module.REMOVE_STOPWORDS)

    print("\nPreprocessing Stage Completed Successfully!")


if __name__ == "__main__":
    run()