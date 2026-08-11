import os
import numpy as np

from datasets import load_dataset
from umap import UMAP
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    ENGLISH_STOP_WORDS
)

from sklearn.decomposition import (
    TruncatedSVD,
    LatentDirichletAllocation
)

from bertopic import BERTopic

import config


# =========================================================
# CUSTOM STOP WORDS
# =========================================================

CUSTOM_STOP_WORDS = {
    "feel",
    "feeling",
    "feels",
    "felt",
    "im",
    "ive",
    "id",
    "dont",
    "didnt",
    "doesnt",
    "cant",
    "couldnt",
    "like",
    "just",
    "really",
    "little",
    "bit",
    "know",
    "want",
    "think",
    "time",
    "things",
    "thing",
    "people",
    "way"
}


# =========================================================
# BUILD COMPLETE STOP WORD LIST
# =========================================================

def get_stop_words():

    stop_words = set(
        ENGLISH_STOP_WORDS
    )

    stop_words.update(
        CUSTOM_STOP_WORDS
    )

    return sorted(
        stop_words
    )


# =========================================================
# PRINT TOPIC WORDS
# =========================================================

def print_topic_words(
    title,
    topic_words
):

    print(
        "\n"
        + "=" * 60
    )

    print(title)

    print(
        "=" * 60
    )

    for topic_id, words in (
        topic_words.items()
    ):

        print(
            f"\nTopic {topic_id}:"
        )

        print(
            ", ".join(words)
        )


# =========================================================
# LSA
# TF-IDF + TRUNCATED SVD
# =========================================================

def run_lsa(
    texts,
    num_topics,
    top_words,
    stop_words
):

    print(
        "\nRunning improved LSA..."
    )

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = (
        vectorizer.fit_transform(
            texts
        )
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    svd = TruncatedSVD(
        n_components=num_topics,
        random_state=42
    )

    svd.fit(
        tfidf_matrix
    )

    topic_words = {}

    for topic_index, component in enumerate(
        svd.components_
    ):

        top_indices = (
            component
            .argsort()[-top_words:]
            [::-1]
        )

        words = [
            feature_names[index]
            for index in top_indices
        ]

        topic_words[
            topic_index
        ] = words

    return (
        topic_words,
        tfidf_matrix,
        svd,
        vectorizer
    )


# =========================================================
# LDA
# COUNT VECTORS + LDA
# =========================================================

def run_lda(
    texts,
    num_topics,
    top_words,
    stop_words
):

    print(
        "\nRunning improved LDA..."
    )

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    count_matrix = (
        vectorizer.fit_transform(
            texts
        )
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        learning_method="batch"
    )

    lda.fit(
        count_matrix
    )

    topic_words = {}

    for topic_index, component in enumerate(
        lda.components_
    ):

        top_indices = (
            component
            .argsort()[-top_words:]
            [::-1]
        )

        words = [
            feature_names[index]
            for index in top_indices
        ]

        topic_words[
            topic_index
        ] = words

    return (
        topic_words,
        count_matrix,
        lda,
        vectorizer
    )


# =========================================================
# BERTOPIC
# =========================================================

def run_bertopic(
    texts,
    min_topic_size,
    top_words,
    stop_words
):

    print(
        "\nRunning improved BERTopic..."
    )

    # Cleaner c-TF-IDF representation
    bertopic_vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=2
    )
    umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
    )
    topic_model = BERTopic(
        vectorizer_model=bertopic_vectorizer,
        umap_model=umap_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False
    )

    topics, probabilities = (
        topic_model.fit_transform(
            texts
        )
    )

    topic_info = (
        topic_model.get_topic_info()
    )

    topic_words = {}

    for topic_id in topic_info[
        "Topic"
    ].tolist():

        # BERTopic uses -1 for outliers
        if topic_id == -1:
            continue

        topic_terms = (
            topic_model.get_topic(
                topic_id
            )
        )

        if not topic_terms:
            continue

        words = [
            word
            for word, score
            in topic_terms[
                :top_words
            ]
        ]

        topic_words[
            topic_id
        ] = words

    return (
        topic_words,
        topics,
        topic_model
    )


# =========================================================
# FAIR COHERENCE CALCULATION
# =========================================================

def calculate_topic_coherence(
    texts,
    topic_words
):
    """
    Document co-occurrence coherence supporting
    both unigrams and bigrams.

    Higher score means that top topic terms
    tend to occur together in documents.
    """

    processed_documents = []

    for text in texts:

        words = text.lower().split()

        # Unigrams
        terms = set(words)

        # Bigrams
        bigrams = {
            words[i] + " " + words[i + 1]
            for i in range(len(words) - 1)
        }

        terms.update(bigrams)

        processed_documents.append(
            terms
        )

    topic_scores = []

    for words in topic_words.values():

        if len(words) < 2:
            continue

        pair_scores = []

        for i in range(len(words)):

            for j in range(
                i + 1,
                len(words)
            ):

                word_1 = words[i]
                word_2 = words[j]

                count_word_1 = 0
                count_word_2 = 0
                count_both = 0

                for document in processed_documents:

                    has_word_1 = (
                        word_1 in document
                    )

                    has_word_2 = (
                        word_2 in document
                    )

                    if has_word_1:
                        count_word_1 += 1

                    if has_word_2:
                        count_word_2 += 1

                    if (
                        has_word_1
                        and has_word_2
                    ):
                        count_both += 1

                # Skip pairs where neither term
                # genuinely occurs in the corpus
                if (
                    count_word_1 == 0
                    or count_word_2 == 0
                ):
                    continue

                score = (
                    count_both + 1
                ) / (
                    min(
                        count_word_1,
                        count_word_2
                    )
                    + 1
                )

                pair_scores.append(
                    score
                )

        if pair_scores:

            topic_scores.append(
                float(
                    np.mean(
                        pair_scores
                    )
                )
            )

    if not topic_scores:
        return 0.0

    return float(
        np.mean(
            topic_scores
        )
    )


# =========================================================
# SIMPLE QUALITATIVE INTERPRETATION
# =========================================================

def qualitative_topic_summary(
    model_name,
    topic_words
):

    print(
        f"\n===== {model_name} "
        "QUALITATIVE CHECK ====="
    )

    number_of_topics = len(
        topic_words
    )

    print(
        "Number of discovered topics:",
        number_of_topics
    )

    if number_of_topics == 0:

        print(
            "No interpretable topics found."
        )

        return

    example_topics = list(
        topic_words.items()
    )[:3]

    for topic_id, words in (
        example_topics
    ):

        print(
            f"Topic {topic_id}:",
            ", ".join(words)
        )


# =========================================================
# SAVE TOPIC WORDS
# =========================================================

def save_topic_words(
    file,
    title,
    topic_words
):

    file.write(
        "\n"
        + "=" * 60
        + "\n"
    )

    file.write(
        title
        + "\n"
    )

    file.write(
        "=" * 60
        + "\n"
    )

    for topic_id, words in (
        topic_words.items()
    ):

        file.write(
            f"\nTopic {topic_id}:\n"
        )

        file.write(
            ", ".join(words)
            + "\n"
        )


# =========================================================
# RUN
# =========================================================

def run(
    config_module=config
):

    print(
        "\nStarting Improved "
        "Topic Modelling Stage..."
    )

    # --------------------------------
    # LOAD DATA
    # --------------------------------

    dataset = load_dataset(
        config_module.DATASET_NAME
    )

    train_texts = dataset[
        "train"
    ][
        config_module.TEXT_COLUMN
    ]

    sample_size = min(
        config_module
        .TOPIC_MODEL_SAMPLE_SIZE,
        len(train_texts)
    )

    texts = train_texts[
        :sample_size
    ]

    print(
        "\nDocuments used:",
        len(texts)
    )

    # --------------------------------
    # STOP WORDS
    # --------------------------------

    stop_words = (
        get_stop_words()
    )

    print(
        "Custom stop words added:",
        len(
            CUSTOM_STOP_WORDS
        )
    )


    # =====================================================
    # LSA
    # =====================================================

    (
        lsa_topics,
        lsa_matrix,
        lsa_model,
        lsa_vectorizer
    ) = run_lsa(
        texts,
        config_module.NUM_TOPICS,
        config_module
        .TOP_WORDS_PER_TOPIC,
        stop_words
    )

    print_topic_words(
        "IMPROVED LSA TOPICS",
        lsa_topics
    )

    qualitative_topic_summary(
        "LSA",
        lsa_topics
    )


    # =====================================================
    # LDA
    # =====================================================

    (
        lda_topics,
        lda_matrix,
        lda_model,
        lda_vectorizer
    ) = run_lda(
        texts,
        config_module.NUM_TOPICS,
        config_module
        .TOP_WORDS_PER_TOPIC,
        stop_words
    )

    print_topic_words(
        "IMPROVED LDA TOPICS",
        lda_topics
    )

    qualitative_topic_summary(
        "LDA",
        lda_topics
    )


    # =====================================================
    # BERTOPIC
    # =====================================================

    (
        bertopic_topics,
        bertopic_assignments,
        bertopic_model
    ) = run_bertopic(
        texts,
        config_module
        .BERTopic_MIN_TOPIC_SIZE,
        config_module
        .TOP_WORDS_PER_TOPIC,
        stop_words
    )

    print_topic_words(
        "IMPROVED BERTOPIC TOPICS",
        bertopic_topics
    )

    qualitative_topic_summary(
        "BERTopic",
        bertopic_topics
    )


    # =====================================================
    # COHERENCE COMPARISON
    # =====================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FAIR COHERENCE COMPARISON"
    )

    print(
        "=" * 60
    )

    lsa_coherence = (
        calculate_topic_coherence(
            texts,
            lsa_topics
        )
    )

    lda_coherence = (
        calculate_topic_coherence(
            texts,
            lda_topics
        )
    )

    bertopic_coherence = (
        calculate_topic_coherence(
            texts,
            bertopic_topics
        )
    )

    coherence_results = {
        "LSA":
            lsa_coherence,

        "LDA":
            lda_coherence,

        "BERTopic":
            bertopic_coherence
    }

    for model_name, score in (
        coherence_results.items()
    ):

        print(
            f"{model_name:<12}"
            f"{score:.4f}"
        )


    # =====================================================
    # NUMERICAL WINNER
    # =====================================================

    numerical_winner = max(
        coherence_results,
        key=coherence_results.get
    )

    print(
        "\nNumerical coherence winner:",
        numerical_winner
    )

    print(
        "Best coherence score:",
        round(
            coherence_results[
                numerical_winner
            ],
            4
        )
    )


    # =====================================================
    # QUALITATIVE CONCLUSION
    # =====================================================

    print(
        "\n===== QUALITATIVE "
        "INTERPRETATION ====="
    )

    print(
        "LSA and LDA should be inspected "
        "for repeated generic words."
    )

    print(
        "BERTopic should be inspected for "
        "clear semantic clusters such as "
        "fear, anger, relationships, "
        "family, books, music, etc."
    )

    print(
        "The final choice should consider "
        "both the coherence score and "
        "human interpretability."
    )


    # =====================================================
    # SAVE REPORT
    # =====================================================

    output_dir = os.path.join(
        config_module.OUTPUT_DIR,
        "topic_modeling"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    report_path = os.path.join(
        output_dir,
        "topic_modeling_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "IMPROVED TOPIC MODELLING "
            "REPORT\n"
        )

        file.write(
            "=" * 60
            + "\n"
        )

        file.write(
            f"Documents used: "
            f"{len(texts)}\n"
        )

        file.write(
            f"Requested LSA/LDA topics: "
            f"{config_module.NUM_TOPICS}\n"
        )

        file.write(
            f"Custom stop words: "
            f"{len(CUSTOM_STOP_WORDS)}\n"
        )

        file.write(
            "N-grams: unigram + bigram\n"
        )

        save_topic_words(
            file,
            "IMPROVED LSA TOPICS",
            lsa_topics
        )

        save_topic_words(
            file,
            "IMPROVED LDA TOPICS",
            lda_topics
        )

        save_topic_words(
            file,
            "IMPROVED BERTOPIC TOPICS",
            bertopic_topics
        )

        file.write(
            "\n"
            + "=" * 60
            + "\n"
        )

        file.write(
            "COHERENCE COMPARISON\n"
        )

        file.write(
            "=" * 60
            + "\n"
        )

        for model_name, score in (
            coherence_results.items()
        ):

            file.write(
                f"{model_name}: "
                f"{score:.4f}\n"
            )

        file.write(
            f"\nNumerical winner: "
            f"{numerical_winner}\n"
        )

        file.write(
            "Best coherence score: "
            f"{coherence_results[numerical_winner]:.4f}\n"
        )

        file.write(
            "\nQUALITATIVE INTERPRETATION\n"
        )

        file.write(
            "The final model should be "
            "selected using both numerical "
            "coherence and human "
            "interpretability.\n"
        )

        file.write(
            "BERTopic is expected to produce "
            "more semantically focused topics "
            "because it uses contextual "
            "sentence embeddings.\n"
        )


    # =====================================================
    # SAVE BERTOPIC MODEL
    # =====================================================

    bertopic_model_path = os.path.join(
        output_dir,
        "bertopic_model_improved"
    )

    bertopic_model.save(
        bertopic_model_path,
        serialization="safetensors",
        save_ctfidf=True
    )

    print(
        "\nSaved report:",
        report_path
    )

    print(
        "Saved BERTopic model:",
        bertopic_model_path
    )

    print(
        "\nImproved Topic Modelling "
        "Stage Completed Successfully!"
    )