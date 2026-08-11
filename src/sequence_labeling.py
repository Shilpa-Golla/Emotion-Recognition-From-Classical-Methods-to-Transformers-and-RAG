import math
import os
from collections import Counter, defaultdict

import nltk
import sklearn_crfsuite

from datasets import load_dataset
from nltk.corpus import brown
from sklearn_crfsuite import metrics

from seqeval.metrics import (
    f1_score as seqeval_f1_score,
    classification_report as seqeval_classification_report
)

import config


# =========================================================
# DOWNLOAD REQUIRED NLTK DATA
# =========================================================

nltk.download("brown", quiet=True)
nltk.download("universal_tagset", quiet=True)


# =========================================================
# PART 1
# HMM POS TAGGER FROM SCRATCH
# =========================================================

class HMMPOSTagger:

    def __init__(self, smoothing=1.0):

        self.smoothing = smoothing

        self.tag_counts = Counter()

        self.transition_counts = defaultdict(
            Counter
        )

        self.emission_counts = defaultdict(
            Counter
        )

        self.tags = []

        self.vocabulary = set()

        self.start_tag = "<START>"
        self.end_tag = "<END>"


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    def fit(self, tagged_sentences):

        for sentence in tagged_sentences:

            previous_tag = self.start_tag

            for word, tag in sentence:

                word = word.lower()

                self.vocabulary.add(word)

                self.tag_counts[tag] += 1

                self.emission_counts[tag][word] += 1

                self.transition_counts[
                    previous_tag
                ][tag] += 1

                previous_tag = tag

            self.transition_counts[
                previous_tag
            ][self.end_tag] += 1

        self.tags = sorted(
            self.tag_counts.keys()
        )

        return self


    # -----------------------------------------------------
    # TRANSITION PROBABILITY
    # -----------------------------------------------------

    def transition_probability(
        self,
        previous_tag,
        current_tag
    ):

        possible_tags = (
            len(self.tags) + 1
        )

        numerator = (
            self.transition_counts[
                previous_tag
            ][current_tag]
            + self.smoothing
        )

        denominator = (
            sum(
                self.transition_counts[
                    previous_tag
                ].values()
            )
            +
            self.smoothing
            * possible_tags
        )

        return numerator / denominator


    # -----------------------------------------------------
    # EMISSION PROBABILITY
    # -----------------------------------------------------

    def emission_probability(
        self,
        tag,
        word
    ):

        word = word.lower()

        vocabulary_size = (
            len(self.vocabulary) + 1
        )

        numerator = (
            self.emission_counts[
                tag
            ][word]
            + self.smoothing
        )

        denominator = (
            self.tag_counts[tag]
            +
            self.smoothing
            * vocabulary_size
        )

        return numerator / denominator


    # -----------------------------------------------------
    # VITERBI
    # -----------------------------------------------------

    def viterbi(self, words):

        if not words:
            return []

        words = [
            word.lower()
            for word in words
        ]

        previous_scores = {}

        # First word
        first_word = words[0]

        for tag in self.tags:

            transition = (
                self.transition_probability(
                    self.start_tag,
                    tag
                )
            )

            emission = (
                self.emission_probability(
                    tag,
                    first_word
                )
            )

            score = (
                math.log(transition)
                +
                math.log(emission)
            )

            previous_scores[tag] = (
                score,
                [tag]
            )

        # Remaining words
        for word in words[1:]:

            current_scores = {}

            for current_tag in self.tags:

                best_score = -float("inf")
                best_path = None

                emission = (
                    self.emission_probability(
                        current_tag,
                        word
                    )
                )

                log_emission = math.log(
                    emission
                )

                for previous_tag, (
                    previous_score,
                    previous_path
                ) in previous_scores.items():

                    transition = (
                        self.transition_probability(
                            previous_tag,
                            current_tag
                        )
                    )

                    score = (
                        previous_score
                        +
                        math.log(transition)
                        +
                        log_emission
                    )

                    if score > best_score:

                        best_score = score

                        best_path = (
                            previous_path
                            +
                            [current_tag]
                        )

                current_scores[
                    current_tag
                ] = (
                    best_score,
                    best_path
                )

            previous_scores = current_scores

        # End transition
        best_final_score = -float("inf")
        best_final_path = None

        for tag, (
            score,
            path
        ) in previous_scores.items():

            end_probability = (
                self.transition_probability(
                    tag,
                    self.end_tag
                )
            )

            final_score = (
                score
                +
                math.log(
                    end_probability
                )
            )

            if final_score > best_final_score:

                best_final_score = (
                    final_score
                )

                best_final_path = path

        return best_final_path


# =========================================================
# HMM EVALUATION
# =========================================================

def evaluate_hmm(
    model,
    tagged_sentences
):

    total = 0
    correct = 0

    example = None

    for sentence in tagged_sentences:

        words = [
            word
            for word, _
            in sentence
        ]

        true_tags = [
            tag
            for _, tag
            in sentence
        ]

        predicted_tags = (
            model.viterbi(
                words
            )
        )

        if example is None:

            example = list(
                zip(
                    words,
                    true_tags,
                    predicted_tags
                )
            )

        for true_tag, predicted_tag in zip(
            true_tags,
            predicted_tags
        ):

            total += 1

            if true_tag == predicted_tag:
                correct += 1

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    return accuracy, example


# =========================================================
# PART 2
# CRF NER
# =========================================================

NER_LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC"
]


# =========================================================
# CRF FEATURES
# =========================================================

def word2features(
    sentence,
    index
):

    word = sentence[index]

    features = {
        "bias": 1.0,

        "word.lower":
            word.lower(),

        "word[-3:]":
            word[-3:],

        "word[-2:]":
            word[-2:],

        "word.isupper":
            word.isupper(),

        "word.istitle":
            word.istitle(),

        "word.isdigit":
            word.isdigit(),

        "word.length":
            len(word)
    }

    # Previous word
    if index > 0:

        previous_word = (
            sentence[index - 1]
        )

        features.update(
            {
                "-1:word.lower":
                    previous_word.lower(),

                "-1:word.istitle":
                    previous_word.istitle(),

                "-1:word.isupper":
                    previous_word.isupper()
            }
        )

    else:
        features["BOS"] = True

    # Next word
    if index < len(sentence) - 1:

        next_word = (
            sentence[index + 1]
        )

        features.update(
            {
                "+1:word.lower":
                    next_word.lower(),

                "+1:word.istitle":
                    next_word.istitle(),

                "+1:word.isupper":
                    next_word.isupper()
            }
        )

    else:
        features["EOS"] = True

    return features


def sentence2features(
    sentence
):

    return [
        word2features(
            sentence,
            index
        )
        for index
        in range(len(sentence))
    ]


# =========================================================
# CONVERT DATASET LABELS TO BIO LABELS
# =========================================================

def convert_ner_labels(
    label_ids
):

    return [
        NER_LABELS[
            label_id
        ]
        for label_id in label_ids
    ]


# =========================================================
# LOAD CONLL2003
# =========================================================

def load_ner_data(
    config_module
):

    print(
        "\nLoading CoNLL-2003 "
        "NER dataset..."
    )

    dataset = load_dataset(
        "eriktks/conll2003",
        revision="refs/convert/parquet"
    )

    train = dataset["train"]
    test = dataset["test"]

    train_size = min(
        config_module.CRF_TRAIN_SENTENCES,
        len(train)
    )

    test_size = min(
        config_module.CRF_TEST_SENTENCES,
        len(test)
    )

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    # Train
    for index in range(
        train_size
    ):

        tokens = train[
            index
        ]["tokens"]

        labels = train[
            index
        ]["ner_tags"]

        X_train.append(
            sentence2features(
                tokens
            )
        )

        y_train.append(
            convert_ner_labels(
                labels
            )
        )

    # Test
    for index in range(
        test_size
    ):

        tokens = test[
            index
        ]["tokens"]

        labels = test[
            index
        ]["ner_tags"]

        X_test.append(
            sentence2features(
                tokens
            )
        )

        y_test.append(
            convert_ner_labels(
                labels
            )
        )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        test
    )


# =========================================================
# RUN
# =========================================================

def run(
    config_module=config
):

    print(
        "\nStarting Sequence "
        "Labelling Stage..."
    )

    # =====================================================
    # HMM POS
    # =====================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "HMM + VITERBI POS "
        "TAGGING FROM SCRATCH"
    )

    print(
        "=" * 60
    )

    tagged_sentences = (
        brown.tagged_sents(
            tagset="universal"
        )
    )

    train_sentences = (
        tagged_sentences[
            :config_module
            .HMM_TRAIN_SENTENCES
        ]
    )

    test_sentences = (
        tagged_sentences[
            config_module
            .HMM_TRAIN_SENTENCES:
            config_module
            .HMM_TRAIN_SENTENCES
            +
            config_module
            .HMM_TEST_SENTENCES
        ]
    )

    print(
        "HMM training sentences:",
        len(train_sentences)
    )

    print(
        "HMM test sentences:",
        len(test_sentences)
    )

    hmm_model = HMMPOSTagger(
        smoothing=(
            config_module
            .HMM_SMOOTHING
        )
    )

    hmm_model.fit(
        train_sentences
    )

    hmm_accuracy, hmm_example = (
        evaluate_hmm(
            hmm_model,
            test_sentences
        )
    )

    print(
        "\nHMM POS Accuracy:",
        round(
            hmm_accuracy,
            4
        )
    )

    print(
        "\n===== VITERBI EXAMPLE ====="
    )

    print(
        f"{'Word':<20}"
        f"{'True':<12}"
        f"{'Predicted':<12}"
    )

    print(
        "-" * 44
    )

    for (
        word,
        true_tag,
        predicted_tag
    ) in hmm_example[:20]:

        print(
            f"{word:<20}"
            f"{true_tag:<12}"
            f"{predicted_tag:<12}"
        )


    # =====================================================
    # CRF NER
    # =====================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CRF NAMED ENTITY "
        "RECOGNITION"
    )

    print(
        "=" * 60
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        test_dataset
    ) = load_ner_data(
        config_module
    )

    print(
        "CRF training sentences:",
        len(X_train)
    )

    print(
        "CRF test sentences:",
        len(X_test)
    )


    # =====================================================
    # CRF FEATURE LIST
    # =====================================================

    print(
        "\n===== CRF FEATURES ====="
    )

    feature_names = [
        "Current word",
        "Lowercase word",
        "Last 2 characters",
        "Last 3 characters",
        "Is uppercase?",
        "Is title case?",
        "Is digit?",
        "Word length",
        "Previous word",
        "Previous word case",
        "Next word",
        "Next word case",
        "Beginning of sentence",
        "End of sentence"
    ]

    for feature in feature_names:
        print("-", feature)


    # =====================================================
    # TRAIN CRF
    # =====================================================

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",

        c1=config_module.CRF_C1,

        c2=config_module.CRF_C2,

        max_iterations=(
            config_module
            .CRF_MAX_ITERATIONS
        ),

        all_possible_transitions=True
    )

    print(
        "\nTraining CRF..."
    )

    crf.fit(
        X_train,
        y_train
    )

    predictions = (
        crf.predict(
            X_test
        )
    )


    # =====================================================
    # TOKEN-LEVEL METRICS
    # =====================================================

    labels = [
        label
        for label
        in crf.classes_
        if label != "O"
    ]

    token_weighted_f1 = (
        metrics.flat_f1_score(
            y_test,
            predictions,
            average="weighted",
            labels=labels
        )
    )

    token_macro_f1 = (
        metrics.flat_f1_score(
            y_test,
            predictions,
            average="macro",
            labels=labels
        )
    )

    print(
        "\nToken Weighted F1:",
        round(
            token_weighted_f1,
            4
        )
    )

    print(
        "Token Macro-F1:",
        round(
            token_macro_f1,
            4
        )
    )

    print(
        "\n===== TOKEN-LEVEL "
        "CLASSIFICATION REPORT ====="
    )

    print(
        metrics.flat_classification_report(
            y_test,
            predictions,
            labels=labels,
            digits=4
        )
    )


    # =====================================================
    # STRICT ENTITY-LEVEL F1
    # =====================================================

    strict_entity_f1 = (
        seqeval_f1_score(
            y_test,
            predictions
        )
    )

    strict_entity_report = (
        seqeval_classification_report(
            y_test,
            predictions,
            digits=4
        )
    )

    print(
        "\nStrict Entity-Level F1:",
        round(
            strict_entity_f1,
            4
        )
    )

    print(
        "\n===== STRICT ENTITY-LEVEL "
        "REPORT ====="
    )

    print(
        strict_entity_report
    )


    # =====================================================
    # EXAMPLE NER PREDICTION
    # =====================================================

    example_tokens = (
        test_dataset[
            0
        ]["tokens"]
    )

    example_true = (
        y_test[0]
    )

    example_predicted = (
        predictions[0]
    )

    print(
        "\n===== NER EXAMPLE ====="
    )

    print(
        f"{'Token':<20}"
        f"{'True':<15}"
        f"{'Predicted':<15}"
    )

    print(
        "-" * 50
    )

    for (
        token,
        true_label,
        predicted_label
    ) in zip(
        example_tokens,
        example_true,
        example_predicted
    ):

        print(
            f"{token:<20}"
            f"{true_label:<15}"
            f"{predicted_label:<15}"
        )


    # =====================================================
    # SAVE REPORT
    # =====================================================

    output_dir = os.path.join(
        config_module.OUTPUT_DIR,
        "sequence_labeling"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    report_path = os.path.join(
        output_dir,
        "sequence_labeling_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SEQUENCE LABELLING REPORT\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "HMM + VITERBI POS TAGGING\n"
        )

        file.write(
            "Implemented from scratch\n"
        )

        file.write(
            f"POS Accuracy: "
            f"{hmm_accuracy:.4f}\n\n"
        )

        file.write(
            "CRF NAMED ENTITY "
            "RECOGNITION\n"
        )

        file.write(
            f"Token Weighted F1: "
            f"{token_weighted_f1:.4f}\n"
        )

        file.write(
            f"Token Macro-F1: "
            f"{token_macro_f1:.4f}\n"
        )

        file.write(
            f"Strict Entity-Level F1: "
            f"{strict_entity_f1:.4f}\n\n"
        )

        file.write(
            "STRICT ENTITY-LEVEL REPORT\n"
        )

        file.write(
            strict_entity_report
        )

        file.write(
            "\n\nCRF FEATURES\n"
        )

        for feature in feature_names:
            file.write(
                f"- {feature}\n"
            )

    print(
        "\nSaved:",
        report_path
    )

    print(
        "\nSequence Labelling Stage "
        "Completed Successfully!"
    )