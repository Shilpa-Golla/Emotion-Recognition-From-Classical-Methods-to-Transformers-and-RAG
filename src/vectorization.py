import math
import os
from collections import Counter

import numpy as np

from datasets import load_dataset
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

import config


# -----------------------------------
# TOKENIZER
# -----------------------------------

def tokenize(text):
    return text.lower().split()


# -----------------------------------
# BAG OF WORDS FROM SCRATCH
# -----------------------------------

def bag_of_words(texts):

    vocabulary = {}

    for text in texts:

        for word in tokenize(text):

            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    matrix = np.zeros(
        (
            len(texts),
            len(vocabulary)
        ),
        dtype=np.float32
    )

    for doc_index, text in enumerate(texts):

        counts = Counter(
            tokenize(text)
        )

        for word, count in counts.items():

            if word in vocabulary:

                word_index = vocabulary[word]

                matrix[
                    doc_index,
                    word_index
                ] = count

    return matrix, vocabulary


# -----------------------------------
# TERM FREQUENCY
# -----------------------------------

def calculate_tf(tokens):
    """
    Raw term frequency:
    number of times a term appears
    inside one document.
    """

    return Counter(tokens)


# -----------------------------------
# DOCUMENT FREQUENCY
# -----------------------------------

def document_frequency(texts):

    df = Counter()

    for text in texts:

        unique_words = set(
            tokenize(text)
        )

        for word in unique_words:
            df[word] += 1

    return df


# -----------------------------------
# TF-IDF FROM SCRATCH
# -----------------------------------

def tfidf_from_scratch(texts):

    number_of_documents = len(texts)

    df = document_frequency(
        texts
    )

    vocabulary = sorted(
        df.keys()
    )

    word_to_index = {
        word: index
        for index, word
        in enumerate(vocabulary)
    }

    matrix = np.zeros(
        (
            number_of_documents,
            len(vocabulary)
        ),
        dtype=np.float32
    )

    for doc_index, text in enumerate(texts):

        tokens = tokenize(text)

        tf_values = calculate_tf(
            tokens
        )

        for word, tf in tf_values.items():

            # Formula required in project:
            # TF-IDF(t,d) =
            # TF(t,d) * log(N / df(t))

            idf = math.log(
                number_of_documents
                / df[word]
            )

            tfidf = tf * idf

            word_index = (
                word_to_index[word]
            )

            matrix[
                doc_index,
                word_index
            ] = tfidf

    return (
        matrix,
        vocabulary,
        word_to_index
    )


# -----------------------------------
# SCIKIT-LEARN TF-IDF
# -----------------------------------

def sklearn_tfidf(texts):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        tokenizer=str.split,
        token_pattern=None,
        norm=None,
        use_idf=True,
        smooth_idf=False,
        sublinear_tf=False
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    return (
        matrix,
        vectorizer
    )


# -----------------------------------
# VALIDATE SCRATCH TF-IDF
# -----------------------------------

def validate_tfidf(
    texts,
    scratch_matrix,
    scratch_word_to_index
):

    vectorizer = CountVectorizer(
        lowercase=True,
        tokenizer=str.split,
        token_pattern=None
    )

    count_matrix = (
        vectorizer.fit_transform(
            texts
        )
    )

    number_of_documents = len(texts)

    document_frequencies = np.asarray(
        (count_matrix > 0).sum(
            axis=0
        )
    ).ravel()

    print(
        "\n===== TF-IDF VALIDATION ====="
    )

    # Use words that actually occur
    # in document 0
    first_document_tokens = list(
        dict.fromkeys(
            tokenize(texts[0])
        )
    )

    differences = []

    for word in first_document_tokens:

        if word not in vectorizer.vocabulary_:
            continue

        sklearn_index = (
            vectorizer.vocabulary_[
                word
            ]
        )

        scratch_index = (
            scratch_word_to_index[
                word
            ]
        )

        # Raw TF from sklearn
        tf = count_matrix[
            0,
            sklearn_index
        ]

        # Document frequency
        df = document_frequencies[
            sklearn_index
        ]

        # Same formula used in
        # our scratch implementation
        library_based_value = (
            tf
            * math.log(
                number_of_documents
                / df
            )
        )

        scratch_value = (
            scratch_matrix[
                0,
                scratch_index
            ]
        )

        difference = abs(
            float(scratch_value)
            - float(
                library_based_value
            )
        )

        differences.append(
            difference
        )

        print(
            f"{word:<15}"
            f"Scratch: "
            f"{scratch_value:.6f}   "
            f"Library check: "
            f"{library_based_value:.6f}   "
            f"Diff: "
            f"{difference:.8f}"
        )

    if differences:

        max_difference = max(
            differences
        )

        print(
            "\nMaximum difference:",
            f"{max_difference:.10f}"
        )

        if max_difference < 1e-5:

            print(
                "VALIDATION PASSED"
            )

        else:

            print(
                "VALIDATION FAILED"
            )


# -----------------------------------
# BM25
# -----------------------------------

def bm25_score(
    query,
    documents,
    k1,
    b
):

    tokenized_docs = [
        tokenize(doc)
        for doc in documents
    ]

    doc_lengths = [
        len(doc)
        for doc in tokenized_docs
    ]

    average_doc_length = (
        sum(doc_lengths)
        / len(doc_lengths)
    )

    df = Counter()

    for doc in tokenized_docs:

        for word in set(doc):
            df[word] += 1

    number_of_documents = len(
        documents
    )

    query_terms = tokenize(
        query
    )

    scores = []

    for doc_index, doc in enumerate(
        tokenized_docs
    ):

        counts = Counter(doc)

        score = 0.0

        for term in query_terms:

            if term not in df:
                continue

            idf = math.log(
                1
                +
                (
                    number_of_documents
                    - df[term]
                    + 0.5
                )
                /
                (
                    df[term]
                    + 0.5
                )
            )

            frequency = counts[
                term
            ]

            numerator = (
                frequency
                * (k1 + 1)
            )

            denominator = (
                frequency
                +
                k1
                *
                (
                    1
                    -
                    b
                    +
                    b
                    *
                    (
                        doc_lengths[
                            doc_index
                        ]
                        /
                        average_doc_length
                    )
                )
            )

            score += (
                idf
                *
                (
                    numerator
                    /
                    denominator
                )
            )

        scores.append(
            score
        )

    return scores


# -----------------------------------
# TOP TF-IDF TERMS PER CLASS
# -----------------------------------

def top_terms_per_class(
    texts,
    labels,
    label_names,
    top_n
):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    matrix = (
        vectorizer.fit_transform(
            texts
        )
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    labels_array = np.array(
        labels
    )

    results = {}

    for label_id, label_name in enumerate(
        label_names
    ):

        indices = np.where(
            labels_array
            == label_id
        )[0]

        class_matrix = matrix[
            indices
        ]

        mean_scores = np.asarray(
            class_matrix.mean(
                axis=0
            )
        ).ravel()

        top_indices = (
            mean_scores
            .argsort()[-top_n:]
            [::-1]
        )

        top_words = [
            (
                feature_names[index],
                float(
                    mean_scores[index]
                )
            )
            for index in top_indices
        ]

        results[
            label_name
        ] = top_words

    return results


# -----------------------------------
# RUN
# -----------------------------------

def run(config_module=config):

    print(
        "\nStarting Vectorization Stage..."
    )

    dataset = load_dataset(
        config_module.DATASET_NAME
    )

    train_data = dataset[
        "train"
    ]

    sample_size = min(
        config_module
        .VECTORIZATION_MAX_DOCS,
        len(train_data)
    )

    texts = train_data[
        config_module.TEXT_COLUMN
    ][:sample_size]

    labels = train_data[
        config_module.LABEL_COLUMN
    ][:sample_size]

    print(
        "\nDocuments used:",
        len(texts)
    )

    # --------------------------------
    # BAG OF WORDS
    # --------------------------------

    print(
        "\n===== BAG OF WORDS ====="
    )

    bow_matrix, bow_vocab = (
        bag_of_words(
            texts[:100]
        )
    )

    print(
        "BoW matrix shape:",
        bow_matrix.shape
    )

    print(
        "BoW vocabulary size:",
        len(bow_vocab)
    )

    # --------------------------------
    # TF-IDF FROM SCRATCH
    # --------------------------------

    print(
        "\n===== TF-IDF FROM SCRATCH ====="
    )

    (
        scratch_matrix,
        scratch_vocab,
        scratch_word_to_index
    ) = tfidf_from_scratch(
        texts
    )

    print(
        "Scratch TF-IDF shape:",
        scratch_matrix.shape
    )

    # --------------------------------
    # SCIKIT-LEARN TF-IDF
    # --------------------------------

    print(
        "\n===== SCIKIT-LEARN TF-IDF ====="
    )

    (
        sklearn_matrix,
        sklearn_vectorizer
    ) = sklearn_tfidf(
        texts
    )

    print(
        "Scikit-learn TF-IDF shape:",
        sklearn_matrix.shape
    )

    # --------------------------------
    # VALIDATION
    # --------------------------------

    validate_tfidf(
        texts,
        scratch_matrix,
        scratch_word_to_index
    )

    # --------------------------------
    # BM25
    # --------------------------------

    print(
        "\n===== BM25 EXAMPLE ====="
    )

    example_documents = [
        "I feel very happy today",
        "I am sad and lonely",
        "I love spending time with my family",
        "I am afraid of the dark"
    ]

    query = "happy feeling"

    scores = bm25_score(
        query,
        example_documents,
        config_module.BM25_K1,
        config_module.BM25_B
    )

    print(
        "Query:",
        query
    )

    for document, score in zip(
        example_documents,
        scores
    ):

        print(
            f"{score:.4f} -> "
            f"{document}"
        )

    # --------------------------------
    # TOP TERMS PER CLASS
    # --------------------------------

    print(
        "\n===== TOP TF-IDF TERMS "
        "PER CLASS ====="
    )

    class_terms = (
        top_terms_per_class(
            texts,
            labels,
            config_module
            .LABEL_NAMES,
            config_module
            .TOP_TFIDF_TERMS
        )
    )

    for label_name, terms in (
        class_terms.items()
    ):

        print(
            f"\n{label_name.upper()}:"
        )

        for word, score in terms:

            print(
                f"{word:<15} "
                f"{score:.4f}"
            )

    # --------------------------------
    # SAVE REPORT
    # --------------------------------

    os.makedirs(
        config_module.OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        config_module.OUTPUT_DIR,
        "vectorization_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "VECTORIZATION REPORT\n"
        )

        file.write(
            f"Documents used: "
            f"{len(texts)}\n"
        )

        file.write(
            f"Scratch TF-IDF shape: "
            f"{scratch_matrix.shape}\n"
        )

        file.write(
            f"Sklearn TF-IDF shape: "
            f"{sklearn_matrix.shape}\n"
        )

        file.write(
            "\nTOP TF-IDF TERMS "
            "PER CLASS\n"
        )

        for label_name, terms in (
            class_terms.items()
        ):

            file.write(
                f"\n{label_name.upper()}\n"
            )

            for word, score in terms:

                file.write(
                    f"{word}: "
                    f"{score:.4f}\n"
                )

    print(
        "\nSaved:",
        output_path
    )

    print(
        "\nVectorization Stage "
        "Completed Successfully!"
    )