# =========================
# PROJECT CONFIGURATION
# =========================

# Dataset
DATASET_NAME = "dair-ai/emotion"

# Reproducibility
SEED = 42

# Output folders
OUTPUT_DIR = "outputs"
DATA_DIR = "data"

# Dataset columns
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Emotion labels
LABEL_NAMES = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

# EDA settings
TOP_N_WORDS = 20
MAX_TEXT_LENGTH_PLOT = 100

# Preprocessing settings
LOWERCASE = True
REMOVE_PUNCTUATION = True
REMOVE_NUMBERS = False
REMOVE_URLS = True
REMOVE_EMAILS = True
REMOVE_HTML = True
NORMALIZE_WHITESPACE = True
NORMALIZE_UNICODE = True
EXPAND_CONTRACTIONS = True
HANDLE_EMOJIS = True
REMOVE_STOPWORDS = False

# Classical ML
MAX_FEATURES = 20000

# N-gram Language Model
NGRAM_MAX_ORDER = 3

# Transformer settings
TRANSFORMER_MODEL = "distilbert-base-uncased"
TRANSFORMER_MAX_LENGTH = 128
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 3
LEARNING_RATE = 2e-5

# Tokenization settings
BPE_VOCAB_SIZE = 5000
BPE_MIN_FREQUENCY = 2

# Stemming / Lemmatization settings
STEM_LEMMA_SAMPLE_SIZE = 5000

# Semantics settings
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
SEMANTICS_WORD = "car"

# Vectorization settings
VECTORIZATION_MAX_DOCS = 3000
TOP_TFIDF_TERMS = 10
BM25_K1 = 1.5
BM25_B = 0.75

# =========================
# N-GRAM LANGUAGE MODEL
# =========================

NGRAM_SAMPLE_SIZE = 5000
NGRAM_VALIDATION_SIZE = 1000
NGRAM_MAX_ORDER = 3
NGRAM_GENERATION_LENGTH = 15
NGRAM_NUM_SENTENCES = 3

# Absolute discount used by advanced backoff
NGRAM_DISCOUNT = 0.75

# =========================
# CLASSICAL ML SETTINGS
# =========================

CLASSICAL_MAX_FEATURES = 10000
NB_ALPHA = 1.0
LOGISTIC_MAX_ITER = 1000
SVM_C = 1.0

# =========================
# SEQUENCE LABELLING
# =========================

HMM_TRAIN_SENTENCES = 3000
HMM_TEST_SENTENCES = 500
HMM_SMOOTHING = 1.0

CRF_TRAIN_SENTENCES = 3000
CRF_TEST_SENTENCES = 500
CRF_C1 = 0.1
CRF_C2 = 0.1
CRF_MAX_ITERATIONS = 100

# =========================
# TOPIC MODELLING
# =========================

TOPIC_MODEL_SAMPLE_SIZE = 5000
NUM_TOPICS = 6
TOP_WORDS_PER_TOPIC = 10
BERTopic_MIN_TOPIC_SIZE = 30

# =========================
# WORD EMBEDDINGS
# =========================

EMBEDDING_SAMPLE_SIZE = 5000

# Scratch Skip-Gram
WORD2VEC_DIM = 50
WORD2VEC_WINDOW = 2
WORD2VEC_NEGATIVE = 5
WORD2VEC_EPOCHS = 3
WORD2VEC_MIN_COUNT = 2
WORD2VEC_LEARNING_RATE = 0.025
WORD2VEC_MAX_VOCAB = 5000
WORD2VEC_MAX_PAIRS = 100000
WORD2VEC_BATCH_SIZE = 512

# Gensim Word2Vec
GENSIM_VECTOR_SIZE = 100
GENSIM_WINDOW = 5
GENSIM_MIN_COUNT = 2
GENSIM_EPOCHS = 5

# FastText
FASTTEXT_VECTOR_SIZE = 100
FASTTEXT_WINDOW = 5
FASTTEXT_MIN_COUNT = 2
FASTTEXT_EPOCHS = 5

# Pretrained GloVe
GLOVE_MODEL = "glove-wiki-gigaword-50"

# Contextual model
CONTEXTUAL_MODEL = "bert-base-uncased"

# =========================
# TRANSFORMER / DISTILBERT
# =========================

TRANSFORMER_MODEL = "distilbert-base-uncased"

TRANSFORMER_MAX_LENGTH = 128
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 2
TRANSFORMER_LEARNING_RATE = 2e-5

# -------------------------------------------------
# FAST MODE
# -------------------------------------------------
# True  = test that the complete pipeline works
# False = final project training
# -------------------------------------------------

TRANSFORMER_FAST_MODE = False

TRANSFORMER_FAST_TRAIN_SIZE = 3000
TRANSFORMER_FAST_VAL_SIZE = 500
TRANSFORMER_FAST_TEST_SIZE = 500

# ============================================================
# RAG CONFIGURATION
# ============================================================

# Dense semantic retrieval model
RAG_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Generator model
RAG_GENERATOR_MODEL = "google/flan-t5-small"

# Number of documents returned
RAG_TOP_K = 3

# Hybrid retrieval weights
RAG_SPARSE_WEIGHT = 0.5
RAG_DENSE_WEIGHT = 0.5

# Maximum generated answer length
RAG_MAX_NEW_TOKENS = 80

# ------------------------------------------------------------
# RETRIEVAL CONFIDENCE THRESHOLDS
# ------------------------------------------------------------
# Answer only when at least one retrieval signal is strong.
#
# If BOTH:
# sparse < 0.05
# AND
# dense < 0.35
#
# then the system refuses to generate an unsupported answer.
# ------------------------------------------------------------

RAG_MIN_SPARSE_SCORE = 0.05
RAG_MIN_DENSE_SCORE = 0.35
# ============================================================
# FINAL EVALUATION
# ============================================================

FINAL_EVALUATION_DIR = "outputs/final_evaluation"