import os
from collections import Counter

import matplotlib.pyplot as plt
from datasets import load_dataset

import config


def load_emotion_dataset():
    dataset = load_dataset(config.DATASET_NAME)
    return dataset


def print_dataset_info(dataset):
    print("\n===== DATASET INFORMATION =====")
    print(dataset)

    print("\nTrain size:", len(dataset["train"]))
    print("Validation size:", len(dataset["validation"]))
    print("Test size:", len(dataset["test"]))

    print("\nFirst training example:")
    print(dataset["train"][0])


def show_class_distribution(dataset):
    labels = dataset["train"][config.LABEL_COLUMN]

    counts = Counter(labels)

    class_names = []
    class_counts = []

    print("\n===== CLASS DISTRIBUTION =====")

    for label_id in sorted(counts.keys()):
        label_name = config.LABEL_NAMES[label_id]
        count = counts[label_id]

        class_names.append(label_name)
        class_counts.append(count)

        print(f"{label_name}: {count}")

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, class_counts)

    plt.title("Training Set Class Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Examples")

    plt.tight_layout()

    output_path = os.path.join(
        config.OUTPUT_DIR,
        "class_distribution.png"
    )

    plt.savefig(output_path)
    plt.close()

    print("Saved:", output_path)


def show_length_distribution(dataset):
    texts = dataset["train"][config.TEXT_COLUMN]

    lengths = []

    for text in texts:
        number_of_words = len(text.split())
        lengths.append(number_of_words)

    average_length = sum(lengths) / len(lengths)

    print("\n===== TEXT LENGTH STATISTICS =====")
    print("Average sentence length:", round(average_length, 2))
    print("Shortest sentence:", min(lengths))
    print("Longest sentence:", max(lengths))

    plt.figure(figsize=(8, 5))

    plt.hist(lengths, bins=30)

    plt.title("Training Text Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")

    plt.tight_layout()

    output_path = os.path.join(
        config.OUTPUT_DIR,
        "length_distribution.png"
    )

    plt.savefig(output_path)
    plt.close()

    print("Saved:", output_path)


def calculate_vocabulary(dataset):
    texts = dataset["train"][config.TEXT_COLUMN]

    vocabulary = set()

    for text in texts:
        words = text.lower().split()

        for word in words:
            vocabulary.add(word)

    print("\n===== VOCABULARY =====")
    print("Vocabulary size:", len(vocabulary))


def create_zipf_plot(dataset):
    texts = dataset["train"][config.TEXT_COLUMN]

    word_counter = Counter()

    for text in texts:
        words = text.lower().split()
        word_counter.update(words)

    frequencies = sorted(
        word_counter.values(),
        reverse=True
    )

    ranks = range(1, len(frequencies) + 1)

    plt.figure(figsize=(8, 5))

    plt.loglog(ranks, frequencies)

    plt.title("Zipf Plot")
    plt.xlabel("Word Rank")
    plt.ylabel("Word Frequency")

    plt.tight_layout()

    output_path = os.path.join(
        config.OUTPUT_DIR,
        "zipf_plot.png"
    )

    plt.savefig(output_path)
    plt.close()

    print("Saved:", output_path)


def run(config_module=config):
    os.makedirs(config_module.OUTPUT_DIR, exist_ok=True)

    print("\nStarting Data & EDA Stage...")

    dataset = load_emotion_dataset()

    print_dataset_info(dataset)

    show_class_distribution(dataset)

    show_length_distribution(dataset)

    calculate_vocabulary(dataset)

    create_zipf_plot(dataset)

    print("\nData & EDA Stage Completed Successfully!")

    return dataset


if __name__ == "__main__":
    run()