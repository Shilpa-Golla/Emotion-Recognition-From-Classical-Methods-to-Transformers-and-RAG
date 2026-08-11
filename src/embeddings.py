import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

import gensim.downloader as api

from gensim.models import (
    Word2Vec,
    FastText
)

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModel
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import config


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_all_seeds(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =========================================================
# BASIC TOKENIZATION
# =========================================================

def tokenize(text):

    return (
        text
        .lower()
        .split()
    )


# =========================================================
# PREPARE TOKENIZED CORPUS
# =========================================================

def prepare_corpus(
    texts
):

    return [
        tokenize(text)
        for text in texts
        if text.strip()
    ]


# =========================================================
# BUILD VOCABULARY FOR SCRATCH WORD2VEC
# =========================================================

def build_vocabulary(
    tokenized_texts,
    min_count,
    max_vocab
):

    word_counts = Counter()

    for sentence in tokenized_texts:

        word_counts.update(
            sentence
        )

    filtered_words = [
        (
            word,
            count
        )
        for word, count
        in word_counts.most_common(
            max_vocab
        )
        if count >= min_count
    ]

    word_to_index = {
        word: index
        for index, (
            word,
            count
        )
        in enumerate(
            filtered_words
        )
    }

    index_to_word = {
        index: word
        for word, index
        in word_to_index.items()
    }

    frequencies = np.array(
        [
            word_counts[
                index_to_word[index]
            ]
            for index
            in range(
                len(index_to_word)
            )
        ],
        dtype=np.float64
    )

    return (
        word_to_index,
        index_to_word,
        frequencies
    )


# =========================================================
# CREATE POSITIVE SKIP-GRAM PAIRS
# =========================================================

def create_skipgram_pairs(
    tokenized_texts,
    word_to_index,
    window_size,
    max_pairs
):

    pairs = []

    for sentence in tokenized_texts:

        indexed_sentence = [
            word_to_index[word]
            for word in sentence
            if word in word_to_index
        ]

        for center_position in range(
            len(indexed_sentence)
        ):

            center_word = (
                indexed_sentence[
                    center_position
                ]
            )

            left = max(
                0,
                center_position
                - window_size
            )

            right = min(
                len(indexed_sentence),
                center_position
                + window_size
                + 1
            )

            for context_position in range(
                left,
                right
            ):

                if (
                    context_position
                    ==
                    center_position
                ):
                    continue

                context_word = (
                    indexed_sentence[
                        context_position
                    ]
                )

                pairs.append(
                    (
                        center_word,
                        context_word
                    )
                )

                if len(pairs) >= max_pairs:

                    return pairs

    return pairs


# =========================================================
# NEGATIVE SAMPLING DISTRIBUTION
# =========================================================

def create_negative_distribution(
    frequencies
):

    # Standard Word2Vec-style
    # frequency^0.75 distribution.

    distribution = (
        frequencies ** 0.75
    )

    distribution = (
        distribution
        /
        distribution.sum()
    )

    return torch.tensor(
        distribution,
        dtype=torch.float32
    )


# =========================================================
# SCRATCH SKIP-GRAM MODEL
# =========================================================

class SkipGramNegativeSampling(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
        embedding_dim
    ):

        super().__init__()

        self.input_embeddings = (
            nn.Embedding(
                vocab_size,
                embedding_dim
            )
        )

        self.output_embeddings = (
            nn.Embedding(
                vocab_size,
                embedding_dim
            )
        )

        nn.init.uniform_(
            self.input_embeddings.weight,
            -0.5 / embedding_dim,
            0.5 / embedding_dim
        )

        nn.init.zeros_(
            self.output_embeddings.weight
        )


    def forward(
        self,
        center_words,
        context_words,
        negative_words
    ):

        center_vectors = (
            self.input_embeddings(
                center_words
            )
        )

        positive_vectors = (
            self.output_embeddings(
                context_words
            )
        )

        positive_scores = (
            center_vectors
            *
            positive_vectors
        ).sum(
            dim=1
        )

        negative_vectors = (
            self.output_embeddings(
                negative_words
            )
        )

        negative_scores = torch.bmm(
            negative_vectors,
            center_vectors
            .unsqueeze(2)
        ).squeeze(2)

        positive_loss = (
            -torch
            .nn
            .functional
            .logsigmoid(
                positive_scores
            )
            .mean()
        )

        negative_loss = (
            -torch
            .nn
            .functional
            .logsigmoid(
                -negative_scores
            )
            .mean()
        )

        return (
            positive_loss
            +
            negative_loss
        )


# =========================================================
# TRAIN SCRATCH SKIP-GRAM
# =========================================================

def train_scratch_word2vec(
    tokenized_texts,
    config_module
):

    print(
        "\n===== WORD2VEC SKIP-GRAM "
        "FROM SCRATCH ====="
    )

    (
        word_to_index,
        index_to_word,
        frequencies
    ) = build_vocabulary(
        tokenized_texts,
        config_module
        .WORD2VEC_MIN_COUNT,
        config_module
        .WORD2VEC_MAX_VOCAB
    )

    print(
        "Scratch vocabulary size:",
        len(word_to_index)
    )

    positive_pairs = (
        create_skipgram_pairs(
            tokenized_texts,
            word_to_index,
            config_module
            .WORD2VEC_WINDOW,
            config_module
            .WORD2VEC_MAX_PAIRS
        )
    )

    print(
        "Positive training pairs:",
        len(positive_pairs)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Training device:",
        device
    )

    model = (
        SkipGramNegativeSampling(
            vocab_size=len(
                word_to_index
            ),
            embedding_dim=(
                config_module
                .WORD2VEC_DIM
            )
        )
        .to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_module
        .WORD2VEC_LEARNING_RATE
    )

    negative_distribution = (
        create_negative_distribution(
            frequencies
        )
    )

    batch_size = (
        config_module
        .WORD2VEC_BATCH_SIZE
    )

    for epoch in range(
        config_module
        .WORD2VEC_EPOCHS
    ):

        random.shuffle(
            positive_pairs
        )

        epoch_loss = 0.0
        number_of_batches = 0

        for start in range(
            0,
            len(positive_pairs),
            batch_size
        ):

            batch = positive_pairs[
                start:
                start + batch_size
            ]

            centers = torch.tensor(
                [
                    pair[0]
                    for pair in batch
                ],
                dtype=torch.long,
                device=device
            )

            contexts = torch.tensor(
                [
                    pair[1]
                    for pair in batch
                ],
                dtype=torch.long,
                device=device
            )

            negative_words = (
                torch.multinomial(
                    negative_distribution,
                    num_samples=(
                        len(batch)
                        *
                        config_module
                        .WORD2VEC_NEGATIVE
                    ),
                    replacement=True
                )
                .view(
                    len(batch),
                    config_module
                    .WORD2VEC_NEGATIVE
                )
                .to(device)
            )

            optimizer.zero_grad()

            loss = model(
                centers,
                contexts,
                negative_words
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item()
            )

            number_of_batches += 1

        average_loss = (
            epoch_loss
            /
            max(
                number_of_batches,
                1
            )
        )

        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{config_module.WORD2VEC_EPOCHS}"
            f" - Loss: "
            f"{average_loss:.4f}"
        )

    embeddings = (
        model
        .input_embeddings
        .weight
        .detach()
        .cpu()
        .numpy()
    )

    return (
        model,
        embeddings,
        word_to_index,
        index_to_word
    )


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(
    vector_a,
    vector_b
):

    denominator = (
        np.linalg.norm(
            vector_a
        )
        *
        np.linalg.norm(
            vector_b
        )
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(
            vector_a,
            vector_b
        )
        /
        denominator
    )


# =========================================================
# MOST SIMILAR FOR SCRATCH MODEL
# =========================================================

def scratch_most_similar(
    word,
    embeddings,
    word_to_index,
    index_to_word,
    top_n=5
):

    if word not in word_to_index:

        return []

    query_index = (
        word_to_index[
            word
        ]
    )

    query_vector = (
        embeddings[
            query_index
        ]
    )

    results = []

    for index in range(
        len(index_to_word)
    ):

        if index == query_index:
            continue

        candidate_vector = (
            embeddings[index]
        )

        similarity = cosine_similarity(
            query_vector,
            candidate_vector
        )

        results.append(
            (
                index_to_word[index],
                similarity
            )
        )

    results.sort(
        key=lambda item:
            item[1],
        reverse=True
    )

    return results[:top_n]


# =========================================================
# GENSIM CBOW AND SKIP-GRAM
# =========================================================

def train_gensim_models(
    tokenized_texts,
    config_module
):

    print(
        "\n===== GENSIM CBOW "
        "VS SKIP-GRAM ====="
    )

    cbow_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=(
            config_module
            .GENSIM_VECTOR_SIZE
        ),
        window=(
            config_module
            .GENSIM_WINDOW
        ),
        min_count=(
            config_module
            .GENSIM_MIN_COUNT
        ),
        sg=0,
        workers=1,
        seed=config_module.SEED,
        epochs=(
            config_module
            .GENSIM_EPOCHS
        )
    )

    skipgram_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=(
            config_module
            .GENSIM_VECTOR_SIZE
        ),
        window=(
            config_module
            .GENSIM_WINDOW
        ),
        min_count=(
            config_module
            .GENSIM_MIN_COUNT
        ),
        sg=1,
        workers=1,
        seed=config_module.SEED,
        epochs=(
            config_module
            .GENSIM_EPOCHS
        )
    )

    example_word = "happy"

    if (
        example_word
        in cbow_model.wv
    ):

        print(
            "\nCBOW nearest words "
            "to 'happy':"
        )

        for word, score in (
            cbow_model
            .wv
            .most_similar(
                example_word,
                topn=5
            )
        ):

            print(
                f"{word:<15}"
                f"{score:.4f}"
            )

    if (
        example_word
        in skipgram_model.wv
    ):

        print(
            "\nSkip-Gram nearest "
            "words to 'happy':"
        )

        for word, score in (
            skipgram_model
            .wv
            .most_similar(
                example_word,
                topn=5
            )
        ):

            print(
                f"{word:<15}"
                f"{score:.4f}"
            )

    return (
        cbow_model,
        skipgram_model
    )


# =========================================================
# GLOVE
# =========================================================

def load_glove(
    config_module
):

    print(
        "\n===== PRETRAINED GLOVE ====="
    )

    print(
        "Loading:",
        config_module.GLOVE_MODEL
    )

    glove_model = api.load(
        config_module.GLOVE_MODEL
    )

    print(
        "GloVe vocabulary size:",
        len(
            glove_model.key_to_index
        )
    )

    print(
        "GloVe vector dimension:",
        glove_model.vector_size
    )

    return glove_model


# =========================================================
# KING - MAN + WOMAN
# =========================================================

def glove_analogy(
    glove_model
):

    print(
        "\n===== WORD ANALOGY ====="
    )

    print(
        "king - man + woman ≈ ?"
    )

    results = (
        glove_model.most_similar(
            positive=[
                "king",
                "woman"
            ],
            negative=[
                "man"
            ],
            topn=5
        )
    )

    for word, score in results:

        print(
            f"{word:<15}"
            f"{score:.4f}"
        )

    return results


# =========================================================
# FASTTEXT
# =========================================================

def train_fasttext(
    tokenized_texts,
    config_module
):

    print(
        "\n===== FASTTEXT ====="
    )

    model = FastText(
        vector_size=(
            config_module
            .FASTTEXT_VECTOR_SIZE
        ),
        window=(
            config_module
            .FASTTEXT_WINDOW
        ),
        min_count=(
            config_module
            .FASTTEXT_MIN_COUNT
        ),
        workers=1,
        seed=config_module.SEED
    )

    model.build_vocab(
        corpus_iterable=(
            tokenized_texts
        )
    )

    model.train(
        corpus_iterable=(
            tokenized_texts
        ),
        total_examples=len(
            tokenized_texts
        ),
        epochs=(
            config_module
            .FASTTEXT_EPOCHS
        )
    )

    oov_word = (
        "happytastic"
    )

    exact_vocab = (
        oov_word
        in model
        .wv
        .key_to_index
    )

    oov_vector = (
        model.wv[
            oov_word
        ]
    )

    print(
        "OOV demonstration word:",
        oov_word
    )

    print(
        "Exists as exact "
        "training-vocabulary word:",
        exact_vocab
    )

    print(
        "FastText still produced "
        "a vector:",
        oov_vector.shape
    )

    print(
        "Vector norm:",
        round(
            float(
                np.linalg.norm(
                    oov_vector
                )
            ),
            4
        )
    )

    return (
        model,
        oov_word
    )


# =========================================================
# BERT CONTEXTUAL EMBEDDINGS
# =========================================================

def get_contextual_word_vector(
    sentence,
    target_word,
    tokenizer,
    model,
    device
):

    encoded = tokenizer(
        sentence,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(
            device
        )
        for key, value
        in encoded.items()
    }

    with torch.no_grad():

        output = model(
            **encoded
        )

    input_ids = (
        encoded[
            "input_ids"
        ][0]
        .detach()
        .cpu()
    )

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids
    )

    target_positions = [
        index
        for index, token
        in enumerate(tokens)
        if token == target_word.lower()
    ]

    if not target_positions:

        raise ValueError(
            f"Target word "
            f"'{target_word}' was "
            f"not found as a single "
            f"BERT token."
        )

    position = (
        target_positions[0]
    )

    vector = (
        output
        .last_hidden_state[
            0,
            position
        ]
        .detach()
        .cpu()
        .numpy()
    )

    return (
        vector,
        tokens
    )


def contextual_bert_demo(
    config_module
):

    print(
        "\n===== CONTEXTUAL BERT "
        "EMBEDDINGS ====="
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            config_module
            .CONTEXTUAL_MODEL
        )
    )

    model = (
        AutoModel
        .from_pretrained(
            config_module
            .CONTEXTUAL_MODEL
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(
        device
    )

    model.eval()

    sentence_1 = (
        "I deposited money "
        "in the bank."
    )

    sentence_2 = (
        "I sat beside the "
        "river bank."
    )

    target_word = "bank"

    vector_1, tokens_1 = (
        get_contextual_word_vector(
            sentence_1,
            target_word,
            tokenizer,
            model,
            device
        )
    )

    vector_2, tokens_2 = (
        get_contextual_word_vector(
            sentence_2,
            target_word,
            tokenizer,
            model,
            device
        )
    )

    similarity = cosine_similarity(
        vector_1,
        vector_2
    )

    euclidean_difference = float(
        np.linalg.norm(
            vector_1
            -
            vector_2
        )
    )

    print(
        "Sentence 1:",
        sentence_1
    )

    print(
        "Sentence 2:",
        sentence_2
    )

    print(
        "Same target word:",
        target_word
    )

    print(
        "Vector dimension:",
        vector_1.shape[0]
    )

    print(
        "Cosine similarity "
        "between the two "
        "'bank' vectors:",
        round(
            similarity,
            4
        )
    )

    print(
        "Euclidean difference:",
        round(
            euclidean_difference,
            4
        )
    )

    print(
        "\nDifferent context -> "
        "different BERT vector."
    )

    return (
        vector_1,
        vector_2,
        similarity
    )


# =========================================================
# SENTENCE VECTOR USING WORD EMBEDDINGS
# =========================================================

def average_sentence_vector(
    text,
    keyed_vectors
):

    vectors = []

    for word in tokenize(text):

        if word in keyed_vectors:

            vectors.append(
                keyed_vectors[
                    word
                ]
            )

    if not vectors:

        return np.zeros(
            keyed_vectors.vector_size,
            dtype=np.float32
        )

    return np.mean(
        vectors,
        axis=0
    )


# =========================================================
# EMBEDDINGS + CLASSIFIER
# =========================================================

def evaluate_embedding_classifier(
    glove_model,
    dataset,
    config_module
):

    print(
        "\n===== EMBEDDINGS + CLASSIFIER ====="
    )

    train_texts = dataset[
        "train"
    ][
        config_module.TEXT_COLUMN
    ]

    train_labels = dataset[
        "train"
    ][
        config_module.LABEL_COLUMN
    ]

    test_texts = dataset[
        "test"
    ][
        config_module.TEXT_COLUMN
    ]

    test_labels = dataset[
        "test"
    ][
        config_module.LABEL_COLUMN
    ]

    print(
        "Creating GloVe sentence vectors..."
    )

    X_train = np.vstack(
        [
            average_sentence_vector(
                text,
                glove_model
            )
            for text in train_texts
        ]
    )

    X_test = np.vstack(
        [
            average_sentence_vector(
                text,
                glove_model
            )
            for text in test_texts
        ]
    )

    print(
        "Train embedding matrix:",
        X_train.shape
    )

    print(
        "Test embedding matrix:",
        X_test.shape
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=config_module.SEED,
        class_weight="balanced"
    )

    classifier.fit(
        X_train,
        train_labels
    )

    predictions = classifier.predict(
        X_test
    )

    accuracy = accuracy_score(
        test_labels,
        predictions
    )

    precision = precision_score(
        test_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        test_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        test_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "Macro Precision:",
        round(
            precision,
            4
        )
    )

    print(
        "Macro Recall:",
        round(
            recall,
            4
        )
    )

    print(
        "Macro F1:",
        round(
            macro_f1,
            4
        )
    )

    return {
        "accuracy":
            float(accuracy),

        "macro_precision":
            float(precision),

        "macro_recall":
            float(recall),

        "macro_f1":
            float(macro_f1)
    }


# =========================================================
# COMPARISON TABLE
# =========================================================

def print_comparison_table():

    comparison_rows = [
        (
            "Scratch Word2Vec",
            "Static",
            "No",
            "No",
            "Skip-Gram"
        ),
        (
            "Gensim CBOW",
            "Static",
            "No",
            "No",
            "Predict center word"
        ),
        (
            "Gensim Skip-Gram",
            "Static",
            "No",
            "No",
            "Predict context words"
        ),
        (
            "GloVe",
            "Static",
            "No",
            "No",
            "Global co-occurrence"
        ),
        (
            "FastText",
            "Static",
            "No",
            "Yes",
            "Subword"
        ),
        (
            "BERT",
            "Contextual",
            "Yes",
            "Subword",
            "Transformer"
        )
    ]

    print(
        "\n"
        + "=" * 90
    )

    print(
        "EMBEDDING COMPARISON TABLE"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Model':<22}"
        f"{'Vector':<14}"
        f"{'Context?':<12}"
        f"{'OOV?':<12}"
        f"{'Model Type':<25}"
    )

    print(
        "-" * 90
    )

    for row in comparison_rows:

        print(
            f"{row[0]:<22}"
            f"{row[1]:<14}"
            f"{row[2]:<12}"
            f"{row[3]:<12}"
            f"{row[4]:<25}"
        )

    return comparison_rows


# =========================================================
# RUN
# =========================================================

def run(
    config_module=config
):

    print(
        "\nStarting Word "
        "Embeddings Stage..."
    )

    set_all_seeds(
        config_module.SEED
    )

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
        .EMBEDDING_SAMPLE_SIZE,
        len(train_texts)
    )

    embedding_texts = (
        train_texts[
            :sample_size
        ]
    )

    tokenized_texts = (
        prepare_corpus(
            embedding_texts
        )
    )

    print(
        "\nEmbedding training "
        "sentences:",
        len(tokenized_texts)
    )


    # =====================================================
    # 1. SCRATCH WORD2VEC
    # =====================================================

    (
        scratch_model,
        scratch_embeddings,
        scratch_word_to_index,
        scratch_index_to_word
    ) = train_scratch_word2vec(
        tokenized_texts,
        config_module
    )

    print(
        "\nScratch nearest words "
        "to 'happy':"
    )

    scratch_neighbors = (
        scratch_most_similar(
            "happy",
            scratch_embeddings,
            scratch_word_to_index,
            scratch_index_to_word,
            top_n=5
        )
    )

    for word, score in (
        scratch_neighbors
    ):

        print(
            f"{word:<15}"
            f"{score:.4f}"
        )


    # =====================================================
    # 2. GENSIM CBOW / SKIP-GRAM
    # =====================================================

    (
        cbow_model,
        skipgram_model
    ) = train_gensim_models(
        tokenized_texts,
        config_module
    )


    # =====================================================
    # 3. GLOVE
    # =====================================================

    glove_model = load_glove(
        config_module
    )

    analogy_results = (
        glove_analogy(
            glove_model
        )
    )


    # =====================================================
    # 4. FASTTEXT + OOV
    # =====================================================

    (
        fasttext_model,
        oov_word
    ) = train_fasttext(
        tokenized_texts,
        config_module
    )


    # =====================================================
    # 5. CONTEXTUAL BERT
    # =====================================================

    (
        bert_vector_1,
        bert_vector_2,
        bert_similarity
    ) = contextual_bert_demo(
        config_module
    )


    # =====================================================
    # 6. EMBEDDINGS + CLASSIFIER
    # =====================================================

    embedding_classifier_results = (
        evaluate_embedding_classifier(
            glove_model,
            dataset,
            config_module
        )
    )


    # =====================================================
    # 7. COMPARISON TABLE
    # =====================================================

    comparison_rows = (
        print_comparison_table()
    )


    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    output_dir = os.path.join(
        config_module.OUTPUT_DIR,
        "embeddings"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # --------------------------------
    # SAVE SCRATCH MODEL
    # --------------------------------

    scratch_path = os.path.join(
        output_dir,
        "scratch_word2vec.pt"
    )

    torch.save(
        {
            "model_state_dict":
                scratch_model
                .state_dict(),

            "word_to_index":
                scratch_word_to_index,

            "index_to_word":
                scratch_index_to_word
        },
        scratch_path
    )


    # --------------------------------
    # SAVE GENSIM MODELS
    # --------------------------------

    cbow_path = os.path.join(
        output_dir,
        "gensim_cbow.model"
    )

    cbow_model.save(
        cbow_path
    )

    skipgram_path = os.path.join(
        output_dir,
        "gensim_skipgram.model"
    )

    skipgram_model.save(
        skipgram_path
    )

    fasttext_path = os.path.join(
        output_dir,
        "fasttext.model"
    )

    fasttext_model.save(
        fasttext_path
    )


    # --------------------------------
    # SAVE BERT VECTORS
    # --------------------------------

    np.save(
        os.path.join(
            output_dir,
            "bert_bank_financial.npy"
        ),
        bert_vector_1
    )

    np.save(
        os.path.join(
            output_dir,
            "bert_bank_river.npy"
        ),
        bert_vector_2
    )


    # --------------------------------
    # SAVE REPORT
    # --------------------------------

    report_path = os.path.join(
        output_dir,
        "embeddings_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "WORD EMBEDDINGS REPORT\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "1. WORD2VEC SKIP-GRAM "
            "WITH NEGATIVE SAMPLING\n"
        )

        file.write(
            "Implemented from scratch "
            "using PyTorch embeddings.\n"
        )

        file.write(
            f"Embedding dimension: "
            f"{config_module.WORD2VEC_DIM}\n"
        )

        file.write(
            f"Window size: "
            f"{config_module.WORD2VEC_WINDOW}\n"
        )

        file.write(
            f"Negative samples: "
            f"{config_module.WORD2VEC_NEGATIVE}\n\n"
        )

        file.write(
            "Scratch nearest words "
            "to happy:\n"
        )

        for word, score in (
            scratch_neighbors
        ):

            file.write(
                f"{word}: "
                f"{score:.4f}\n"
            )

        file.write(
            "\n2. GLOVE ANALOGY\n"
        )

        file.write(
            "king - man + woman:\n"
        )

        for word, score in (
            analogy_results
        ):

            file.write(
                f"{word}: "
                f"{score:.4f}\n"
            )

        file.write(
            "\n3. FASTTEXT OOV\n"
        )

        file.write(
            f"OOV word: "
            f"{oov_word}\n"
        )

        file.write(
            "FastText generated "
            "a vector using "
            "subword information.\n"
        )

        file.write(
            "\n4. CONTEXTUAL BERT\n"
        )

        file.write(
            "Word: bank\n"
        )

        file.write(
            "Sentence 1: "
            "I deposited money "
            "in the bank.\n"
        )

        file.write(
            "Sentence 2: "
            "I sat beside the "
            "river bank.\n"
        )

        file.write(
            "Cosine similarity "
            "between contextual "
            "vectors: "
            f"{bert_similarity:.4f}\n"
        )

        file.write(
            "\n5. EMBEDDINGS + "
            "CLASSIFIER\n"
        )

        for metric_name, value in (
            embedding_classifier_results
            .items()
        ):

            file.write(
                f"{metric_name}: "
                f"{value:.4f}\n"
            )

        file.write(
            "\n6. EMBEDDING "
            "COMPARISON\n"
        )

        file.write(
            f"{'Model':<22}"
            f"{'Vector':<14}"
            f"{'Context?':<12}"
            f"{'OOV?':<12}"
            f"{'Model Type'}\n"
        )

        for row in comparison_rows:

            file.write(
                f"{row[0]:<22}"
                f"{row[1]:<14}"
                f"{row[2]:<12}"
                f"{row[3]:<12}"
                f"{row[4]}\n"
            )


    print(
        "\nSaved:",
        report_path
    )

    print(
        "Saved scratch model:",
        scratch_path
    )

    print(
        "Saved Gensim CBOW:",
        cbow_path
    )

    print(
        "Saved Gensim Skip-Gram:",
        skipgram_path
    )

    print(
        "Saved FastText:",
        fasttext_path
    )

    print(
        "\nWord Embeddings Stage "
        "Completed Successfully!"
    )

    return (
        embedding_classifier_results
    )