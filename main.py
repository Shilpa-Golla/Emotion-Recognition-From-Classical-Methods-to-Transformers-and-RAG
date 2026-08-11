import config

import config

from src import data_loader
from src import preprocessing
from src import tokenization
from src import stem_lemma
from src import linguistics
from src import semantics
from src import vectorization
from src import ngram_lm
from src import classical_ml
from src import sequence_labeling
from src import topic_modeling
from src import embeddings
from src import transformer_llm
from src import rag
from src import evaluation

def main():
    print("================================")
    print("     NLP PIPELINE PROJECT")
    print("================================")

    print("\nDataset:", config.DATASET_NAME)
    print("Random seed:", config.SEED)
    print("Classes:", config.LABEL_NAMES)

    # Stage 1
    data_loader.run(config)

    # Stage 2
    preprocessing.run(config)
    # Stage 3
    tokenization.run(config)
    # Stage 4
    stem_lemma.run(config)
    # Stage 5
    linguistics.run(config)
    # Stage 6
    semantics.run(config)
    # Stage 7
    vectorization.run(config)
    # Stage 8
    ngram_lm.run(config)
    # Stage 9
    classical_ml.run(config)
    # Stage 10
    sequence_labeling.run(config)
    # Stage 11
    topic_modeling.run(config)
    # Stage 12
    embeddings.run(config)
    # Stage 13
    transformer_llm.run(config)
    # Stage 14
    rag.run(config)
    # Stage 15
    evaluation.run(config)

if __name__ == "__main__":
    main()