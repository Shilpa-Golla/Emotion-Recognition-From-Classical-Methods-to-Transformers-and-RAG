import math
import os
import random
from collections import Counter, defaultdict

from datasets import load_dataset

from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

import config


# =========================================================
# BASIC TOKENIZATION
# =========================================================

def tokenize(text):
    """
    Simple word tokenizer for the n-gram language model.
    """
    return text.lower().split()


# =========================================================
# BUILD TRAINING VOCABULARY
# =========================================================

def build_vocabulary(texts):
    """
    Build vocabulary only from training data.

    <UNK> is used for words appearing in validation/test
    that were never observed during training.
    """

    vocabulary = {
        "<UNK>",
        "<s>",
        "</s>"
    }

    for text in texts:
        vocabulary.update(
            tokenize(text)
        )

    return vocabulary


# =========================================================
# MAP UNKNOWN WORDS
# =========================================================

def map_unknown_words(tokens, vocabulary):
    """
    Replace words not seen in training with <UNK>.
    """

    return [
        token
        if token in vocabulary
        else "<UNK>"
        for token in tokens
    ]


# =========================================================
# PREPARE ONE SENTENCE
# =========================================================

def prepare_sentence(
    text,
    n,
    vocabulary=None
):
    """
    Standard n-gram padding.

    Bigram:
        <s> sentence </s>

    Trigram:
        <s> <s> sentence </s> </s>
    """

    tokens = tokenize(text)

    if vocabulary is not None:
        tokens = map_unknown_words(
            tokens,
            vocabulary
        )

    if n == 1:
        return tokens

    return (
        ["<s>"] * (n - 1)
        +
        tokens
        +
        ["</s>"] * (n - 1)
    )


# =========================================================
# SCRATCH N-GRAM LANGUAGE MODEL
# =========================================================

class NGramLanguageModel:

    def __init__(
        self,
        n,
        discount=0.75
    ):

        self.n = n

        self.discount = discount

        # Counts for every order
        #
        # Example:
        # counts[1] -> unigram counts
        # counts[2] -> bigram counts
        # counts[3] -> trigram counts

        self.counts = {
            order: Counter()
            for order in range(
                1,
                n + 1
            )
        }

        # Context counts
        self.context_counts = {
            order: Counter()
            for order in range(
                2,
                n + 1
            )
        }

        # Which words have followed
        # a particular context?
        self.continuations = {
            order: defaultdict(set)
            for order in range(
                2,
                n + 1
            )
        }

        self.vocabulary = set()

        self.total_unigrams = 0


    # -----------------------------------------------------
    # TRAIN MODEL
    # -----------------------------------------------------

    def fit(self, texts):

        self.vocabulary = (
            build_vocabulary(texts)
        )

        # --------------------------------
        # Count unigrams
        # --------------------------------

        for text in texts:

            tokens = tokenize(text)

            for token in tokens:

                self.counts[1][
                    (token,)
                ] += 1

                self.total_unigrams += 1

        # --------------------------------
        # Count bigrams/trigrams
        # --------------------------------

        for order in range(
            2,
            self.n + 1
        ):

            for text in texts:

                sentence = prepare_sentence(
                    text,
                    order
                )

                for index in range(
                    len(sentence)
                    - order
                    + 1
                ):

                    ngram = tuple(
                        sentence[
                            index:
                            index + order
                        ]
                    )

                    context = ngram[:-1]

                    word = ngram[-1]

                    self.counts[
                        order
                    ][ngram] += 1

                    self.context_counts[
                        order
                    ][context] += 1

                    self.continuations[
                        order
                    ][context].add(
                        word
                    )

        return self


    # =====================================================
    # LAPLACE PROBABILITY
    # =====================================================

    def laplace_probability(
        self,
        word,
        context=()
    ):

        """
        Add-one / Laplace smoothing.
        """

        vocabulary_size = len(
            self.vocabulary
        )

        # -------------------------
        # UNIGRAM
        # -------------------------

        if len(context) == 0:

            count = self.counts[
                1
            ][
                (word,)
            ]

            return (
                count + 1
            ) / (
                self.total_unigrams
                + vocabulary_size
            )

        # -------------------------
        # N-GRAM
        # -------------------------

        order = len(context) + 1

        ngram = tuple(
            context
        ) + (word,)

        ngram_count = self.counts[
            order
        ][ngram]

        context_count = (
            self.context_counts[
                order
            ][tuple(context)]
        )

        return (
            ngram_count + 1
        ) / (
            context_count
            + vocabulary_size
        )


    # =====================================================
    # ADVANCED METHOD:
    # NORMALIZED ABSOLUTE-DISCOUNT BACKOFF
    # =====================================================

    def backoff_probability(
        self,
        word,
        context
    ):

        """
        Absolute-discount backoff.

        If an n-gram is observed:
            use discounted higher-order probability.

        If it is unseen:
            back off to a shorter context.

        The remaining probability mass is normalized
        before being passed to the lower-order model.

        This avoids the zero-probability problem while
        remaining a proper probability distribution.
        """

        # --------------------------------
        # BASE CASE: UNIGRAM
        # --------------------------------

        if len(context) == 0:

            return self.laplace_probability(
                word,
                ()
            )

        # We only keep as much context
        # as this model supports.

        context = tuple(
            context[
                -(self.n - 1):
            ]
        )

        order = len(context) + 1

        context_count = (
            self.context_counts[
                order
            ][context]
        )

        # --------------------------------
        # UNSEEN CONTEXT
        # --------------------------------

        if context_count == 0:

            return self.backoff_probability(
                word,
                context[1:]
            )

        ngram = (
            context
            + (word,)
        )

        ngram_count = self.counts[
            order
        ][ngram]

        seen_words = (
            self.continuations[
                order
            ][context]
        )

        number_of_seen_types = len(
            seen_words
        )

        # --------------------------------
        # SEEN N-GRAM
        # --------------------------------

        if ngram_count > 0:

            probability = (
                max(
                    ngram_count
                    - self.discount,
                    0.0
                )
                /
                context_count
            )

            return probability

        # --------------------------------
        # UNSEEN N-GRAM:
        # BACK OFF
        # --------------------------------

        # Probability mass left after
        # discounting seen n-grams.

        leftover_mass = (
            self.discount
            * number_of_seen_types
            /
            context_count
        )

        lower_context = (
            context[1:]
        )

        lower_word_probability = (
            self.backoff_probability(
                word,
                lower_context
            )
        )

        # Calculate how much lower-order
        # probability mass belongs to words
        # already seen in this context.

        lower_seen_mass = 0.0

        for seen_word in seen_words:

            lower_seen_mass += (
                self.backoff_probability(
                    seen_word,
                    lower_context
                )
            )

        remaining_lower_mass = (
            1.0
            - lower_seen_mass
        )

        # Safety against floating point
        # or extremely degenerate cases.

        if remaining_lower_mass <= 0:

            return self.laplace_probability(
                word,
                context
            )

        alpha = (
            leftover_mass
            /
            remaining_lower_mass
        )

        probability = (
            alpha
            * lower_word_probability
        )

        return probability


    # =====================================================
    # GET N-GRAMS FROM HELD-OUT TEXT
    # =====================================================

    def get_test_ngrams(
        self,
        texts
    ):

        ngrams = []

        for text in texts:

            sentence = prepare_sentence(
                text,
                self.n,
                self.vocabulary
            )

            if self.n == 1:

                for word in sentence:

                    ngrams.append(
                        (word,)
                    )

            else:

                for index in range(
                    len(sentence)
                    - self.n
                    + 1
                ):

                    ngram = tuple(
                        sentence[
                            index:
                            index + self.n
                        ]
                    )

                    ngrams.append(
                        ngram
                    )

        return ngrams


    # =====================================================
    # LAPLACE PERPLEXITY
    # =====================================================

    def laplace_perplexity(
        self,
        texts
    ):

        test_ngrams = (
            self.get_test_ngrams(
                texts
            )
        )

        if not test_ngrams:
            return float("inf")

        log_probability = 0.0

        for ngram in test_ngrams:

            word = ngram[-1]

            context = ngram[:-1]

            probability = (
                self.laplace_probability(
                    word,
                    context
                )
            )

            log_probability += (
                math.log(
                    probability
                )
            )

        return math.exp(
            -
            log_probability
            /
            len(test_ngrams)
        )


    # =====================================================
    # BACKOFF PERPLEXITY
    # =====================================================

    def backoff_perplexity(
        self,
        texts
    ):

        test_ngrams = (
            self.get_test_ngrams(
                texts
            )
        )

        if not test_ngrams:
            return float("inf")

        log_probability = 0.0

        for ngram in test_ngrams:

            word = ngram[-1]

            context = ngram[:-1]

            probability = (
                self.backoff_probability(
                    word,
                    context
                )
            )

            # Numerical safety

            probability = max(
                probability,
                1e-12
            )

            log_probability += (
                math.log(
                    probability
                )
            )

        return math.exp(
            -
            log_probability
            /
            len(test_ngrams)
        )


    # =====================================================
    # SENTENCE GENERATION
    # =====================================================

    def generate(
        self,
        max_length=15
    ):

        # --------------------------------
        # UNIGRAM
        # --------------------------------

        if self.n == 1:

            candidates = []
            weights = []

            for (
                unigram,
                count
            ) in self.counts[1].items():

                word = unigram[0]

                if word not in {
                    "<s>",
                    "</s>",
                    "<UNK>"
                }:

                    candidates.append(
                        word
                    )

                    weights.append(
                        count
                    )

            if not candidates:
                return ""

            generated = random.choices(
                candidates,
                weights=weights,
                k=max_length
            )

            return " ".join(
                generated
            )

        # --------------------------------
        # BIGRAM / TRIGRAM
        # --------------------------------

        context = tuple(
            ["<s>"]
            * (self.n - 1)
        )

        generated = []

        for _ in range(
            max_length
        ):

            candidates = []
            weights = []

            # Start with observed
            # continuations.

            current_order = (
                len(context) + 1
            )

            current_context = (
                context
            )

            while (
                current_order >= 2
            ):

                observed_words = (
                    self.continuations[
                        current_order
                    ].get(
                        current_context,
                        set()
                    )
                )

                if observed_words:

                    for word in (
                        observed_words
                    ):

                        ngram = (
                            current_context
                            + (word,)
                        )

                        count = (
                            self.counts[
                                current_order
                            ][ngram]
                        )

                        candidates.append(
                            word
                        )

                        weights.append(
                            count
                        )

                    break

                # Back off to shorter
                # context.

                current_context = (
                    current_context[1:]
                )

                current_order -= 1

            # If no higher-order
            # continuation exists,
            # fall back to unigram.

            if not candidates:

                for (
                    unigram,
                    count
                ) in self.counts[
                    1
                ].items():

                    word = unigram[0]

                    if word not in {
                        "<s>",
                        "<UNK>"
                    }:

                        candidates.append(
                            word
                        )

                        weights.append(
                            count
                        )

            if not candidates:
                break

            next_word = random.choices(
                candidates,
                weights=weights,
                k=1
            )[0]

            if next_word == "</s>":
                break

            generated.append(
                next_word
            )

            context = (
                context
                + (next_word,)
            )[
                -(self.n - 1):
            ]

        return " ".join(
            generated
        )


# =========================================================
# NLTK LIBRARY MODEL
# =========================================================

def train_nltk_laplace(
    texts,
    n
):

    """
    Train NLTK's Laplace n-gram model
    for validation against our scratch model.
    """

    tokenized_texts = [
        tokenize(text)
        for text in texts
    ]

    train_data, padded_sents = (
        padded_everygram_pipeline(
            n,
            tokenized_texts
        )
    )

    model = Laplace(
        n
    )

    model.fit(
        train_data,
        padded_sents
    )

    return model


# =========================================================
# NLTK PERPLEXITY
# =========================================================

def nltk_perplexity(
    model,
    texts,
    n
):

    ngrams = []

    for text in texts:

        tokens = tokenize(
            text
        )

        # Map OOV through NLTK
        tokens = list(
            model.vocab.lookup(
                tokens
            )
        )

        if n == 1:

            for word in tokens:

                ngrams.append(
                    (word,)
                )

        else:

            sentence = (
                ["<s>"] * (n - 1)
                +
                tokens
                +
                ["</s>"] * (n - 1)
            )

            for index in range(
                len(sentence)
                - n
                + 1
            ):

                ngram = tuple(
                    sentence[
                        index:
                        index + n
                    ]
                )

                ngram = tuple(
                    model.vocab.lookup(
                        ngram
                    )
                )

                ngrams.append(
                    ngram
                )

    return model.perplexity(
        ngrams
    )


# =========================================================
# VALIDATE SCRATCH AGAINST NLTK
# =========================================================

def validate_against_nltk(
    scratch_model,
    nltk_model,
    training_text,
    n
):

    print(
        "\n===== LIBRARY VALIDATION ====="
    )

    print(
        "Validation library: "
        "NLTK Laplace LM"
    )

    sentence = prepare_sentence(
        training_text,
        n,
        scratch_model.vocabulary
    )

    if n == 1:

        example_ngrams = [
            (word,)
            for word in sentence[:4]
        ]

    else:

        example_ngrams = []

        for index in range(
            min(
                4,
                len(sentence)
                - n
                + 1
            )
        ):

            example_ngrams.append(
                tuple(
                    sentence[
                        index:
                        index + n
                    ]
                )
            )

    differences = []

    for ngram in example_ngrams:

        word = ngram[-1]
        context = ngram[:-1]

        scratch_probability = (
            scratch_model
            .laplace_probability(
                word,
                context
            )
        )

        nltk_probability = (
            nltk_model.score(
                word,
                context
            )
        )

        difference = abs(
            scratch_probability
            - nltk_probability
        )

        differences.append(
            difference
        )

        print(
            f"{ngram}\n"
            f"  Scratch: "
            f"{scratch_probability:.10f}\n"
            f"  NLTK:    "
            f"{nltk_probability:.10f}\n"
            f"  Diff:    "
            f"{difference:.10f}"
        )

    if differences:

        max_difference = max(
            differences
        )

        print(
            "\nMaximum probability "
            "difference:",
            f"{max_difference:.10f}"
        )

        if max_difference < 1e-8:

            print(
                "LIBRARY VALIDATION PASSED"
            )

        else:

            print(
                "Library values differ "
                "slightly; inspect vocabulary "
                "or padding conventions."
            )


# =========================================================
# RUN
# =========================================================

def run(
    config_module=config
):

    print(
        "\nStarting N-gram "
        "Language Model Stage..."
    )

    random.seed(
        config_module.SEED
    )

    # --------------------------------
    # LOAD DATASET
    # --------------------------------

    dataset = load_dataset(
        config_module.DATASET_NAME
    )

    train_texts = dataset[
        "train"
    ][
        config_module.TEXT_COLUMN
    ][
        :config_module
        .NGRAM_SAMPLE_SIZE
    ]

    validation_texts = dataset[
        "validation"
    ][
        config_module.TEXT_COLUMN
    ][
        :config_module
        .NGRAM_VALIDATION_SIZE
    ]

    print(
        "\nTraining sentences:",
        len(train_texts)
    )

    print(
        "Held-out validation sentences:",
        len(validation_texts)
    )

    results = []


    # =====================================================
    # TRAIN UNIGRAM / BIGRAM / TRIGRAM
    # =====================================================

    for n in range(
        1,
        config_module
        .NGRAM_MAX_ORDER
        + 1
    ):

        names = {
            1: "UNIGRAM",
            2: "BIGRAM",
            3: "TRIGRAM"
        }

        print(
            "\n"
            + "=" * 55
        )

        print(
            f"{names[n]} MODEL "
            f"(n={n})"
        )

        print(
            "=" * 55
        )

        # --------------------------------
        # SCRATCH MODEL
        # --------------------------------

        scratch_model = (
            NGramLanguageModel(
                n=n,
                discount=(
                    config_module
                    .NGRAM_DISCOUNT
                )
            )
        )

        scratch_model.fit(
            train_texts
        )

        print(
            "\nVocabulary size:",
            len(
                scratch_model
                .vocabulary
            )
        )

        print(
            f"Unique {n}-grams:",
            len(
                scratch_model
                .counts[n]
            )
        )


        # --------------------------------
        # LAPLACE PERPLEXITY
        # --------------------------------

        scratch_laplace_ppl = (
            scratch_model
            .laplace_perplexity(
                validation_texts
            )
        )

        print(
            "\nScratch Laplace "
            "perplexity:",
            round(
                scratch_laplace_ppl,
                4
            )
        )


        # --------------------------------
        # ADVANCED BACKOFF PERPLEXITY
        # --------------------------------

        scratch_backoff_ppl = (
            scratch_model
            .backoff_perplexity(
                validation_texts
            )
        )

        print(
            "Scratch absolute-discount "
            "backoff perplexity:",
            round(
                scratch_backoff_ppl,
                4
            )
        )


        # --------------------------------
        # NLTK LIBRARY MODEL
        # --------------------------------

        nltk_model = (
            train_nltk_laplace(
                train_texts,
                n
            )
        )

        nltk_ppl = nltk_perplexity(
            nltk_model,
            validation_texts,
            n
        )

        print(
            "NLTK Laplace "
            "perplexity:",
            round(
                nltk_ppl,
                4
            )
        )


        # --------------------------------
        # LIBRARY VALIDATION
        # --------------------------------

        validate_against_nltk(
            scratch_model,
            nltk_model,
            train_texts[0],
            n
        )


        # --------------------------------
        # SENTENCE GENERATION
        # --------------------------------

        print(
            "\n===== GENERATED "
            "SENTENCES ====="
        )

        generated_sentences = []

        for sentence_number in range(
            1,
            config_module
            .NGRAM_NUM_SENTENCES
            + 1
        ):

            sentence = (
                scratch_model
                .generate(
                    max_length=(
                        config_module
                        .NGRAM_GENERATION_LENGTH
                    )
                )
            )

            generated_sentences.append(
                sentence
            )

            print(
                f"{sentence_number}. "
                f"{sentence}"
            )


        # --------------------------------
        # SAVE RESULT FOR REPORT
        # --------------------------------

        results.append(
            {
                "n": n,
                "name": names[n],
                "scratch_laplace":
                    scratch_laplace_ppl,
                "scratch_backoff":
                    scratch_backoff_ppl,
                "nltk_laplace":
                    nltk_ppl,
                "sentences":
                    generated_sentences
            }
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 55
    )

    print(
        "N-GRAM PERPLEXITY SUMMARY"
    )

    print(
        "=" * 55
    )

    print(
        f"{'Model':<12}"
        f"{'Scratch Laplace':<20}"
        f"{'Backoff':<15}"
        f"{'NLTK Laplace':<20}"
    )

    print(
        "-" * 67
    )

    for result in results:

        print(
            f"{result['name']:<12}"
            f"{result['scratch_laplace']:<20.4f}"
            f"{result['scratch_backoff']:<15.4f}"
            f"{result['nltk_laplace']:<20.4f}"
        )


    # =====================================================
    # SAVE REPORT
    # =====================================================

    os.makedirs(
        config_module.OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        config_module.OUTPUT_DIR,
        "ngram_lm_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "N-GRAM LANGUAGE MODEL REPORT\n"
        )

        file.write(
            "=" * 50
            + "\n"
        )

        file.write(
            f"Training sentences: "
            f"{len(train_texts)}\n"
        )

        file.write(
            f"Validation sentences: "
            f"{len(validation_texts)}\n"
        )

        file.write(
            "Advanced smoothing method: "
            "Absolute-discount backoff\n"
        )

        file.write(
            f"Discount: "
            f"{config_module.NGRAM_DISCOUNT}\n"
        )

        for result in results:

            file.write(
                "\n"
                + "=" * 40
                + "\n"
            )

            file.write(
                f"{result['name']} "
                f"(n={result['n']})\n"
            )

            file.write(
                "=" * 40
                + "\n"
            )

            file.write(
                "Scratch Laplace perplexity: "
                f"{result['scratch_laplace']:.4f}\n"
            )

            file.write(
                "Scratch backoff perplexity: "
                f"{result['scratch_backoff']:.4f}\n"
            )

            file.write(
                "NLTK Laplace perplexity: "
                f"{result['nltk_laplace']:.4f}\n"
            )

            file.write(
                "Generated sentences:\n"
            )

            for sentence in (
                result["sentences"]
            ):

                file.write(
                    f"- {sentence}\n"
                )

    print(
        "\nSaved:",
        output_path
    )

    print(
        "\nN-gram Language Model "
        "Stage Completed Successfully!"
    )