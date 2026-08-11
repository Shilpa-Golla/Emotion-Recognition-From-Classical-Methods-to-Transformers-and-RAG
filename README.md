# Emotion Recognition: From Classical NLP to Transformers and RAG

An end-to-end Natural Language Processing project for **emotion recognition and text understanding**, built in Python using the Hugging Face `dair-ai/emotion` dataset.

The project progresses from fundamental NLP techniques and classical machine learning to linguistic analysis, statistical language modelling, sequence labelling, topic modelling, word embeddings, Transformer fine-tuning, Retrieval-Augmented Generation (RAG), and final model evaluation.

## Project Overview

The central task is **emotion classification** across six classes: sadness, joy, love, anger, fear, and surprise.

### Best Result

**DistilBERT achieved the best overall emotion-classification performance:**

| Metric | Score |
|---|---:|
| Accuracy | **92.00%** |
| Macro-F1 | **88.62%** |
| Weighted F1 | **92.24%** |

The strongest classical baseline was **Linear SVM**, with **90.45% accuracy** and **85.89% Macro-F1**.

## Main Dataset

**Hugging Face dataset:** `dair-ai/emotion`

The dataset contains 20,000 labelled text examples:

| Split | Samples |
|---|---:|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

Emotion classes:

- sadness
- joy
- love
- anger
- fear
- surprise

The training split is imbalanced, with `joy` and `sadness` being the largest classes and `surprise` the smallest.

For the sequence-labelling module, CoNLL-2003 is additionally used for Named Entity Recognition because the main emotion dataset does not contain BIO entity labels.

---

## Project Objectives

The project demonstrates and compares multiple NLP approaches:

1. Data loading and exploratory analysis
2. Text preprocessing
3. Tokenization
4. Stemming and lemmatization
5. Linguistic analysis
6. Semantics
7. Vectorization
8. N-gram language modelling
9. Classical machine learning
10. Sequence labelling
11. Topic modelling
12. Word embeddings
13. Transformer fine-tuning
14. Retrieval-Augmented Generation
15. Final evaluation

---

## Project Structure

```text
nlp-pipeline/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── tokenization.py
│   ├── stem_lemma.py
│   ├── linguistics.py
│   ├── semantics.py
│   ├── vectorization.py
│   ├── ngram_lm.py
│   ├── classical_ml.py
│   ├── sequence_labeling.py
│   ├── topic_modeling.py
│   ├── embeddings.py
│   ├── transformer_llm.py
│   ├── rag.py
│   └── evaluation.py
│
└── outputs/
    ├── classical_ml/
    ├── sequence_labeling/
    ├── topic_modeling/
    ├── embeddings/
    ├── transformer/
    ├── rag/
    └── final_evaluation/
```

---

## Environment

Recommended Python version:

```text
Python 3.11
```

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Recommended Way to Finalize `requirements.txt`

Because this project contains several libraries with compiled dependencies, the most reliable final pinned requirements file should be generated from the working Python 3.11 virtual environment:

```powershell
pip freeze > requirements.txt
```

Run that command only after the whole project executes successfully.

---

# Pipeline Stages

## 1. Data and EDA

Loads `dair-ai/emotion` and reports:

- dataset split sizes
- class distribution
- text-length statistics
- vocabulary size
- Zipf-style vocabulary analysis

Observed training distribution:

| Class | Count |
|---|---:|
| sadness | 4666 |
| joy | 5362 |
| love | 1304 |
| anger | 2159 |
| fear | 1937 |
| surprise | 572 |

---

## 2. Preprocessing

Includes reusable preprocessing functions for text normalization and cleaning.

---

## 3. Tokenization

Demonstrates tokenization behaviour and prepares text for later NLP stages.

---

## 4. Stemming and Lemmatization

Compares:

- Porter stemming
- Snowball stemming
- WordNet lemmatization
- spaCy lemmatization

Example:

```text
studies -> studi   (Porter)
studies -> study   (WordNet/spaCy)
```

Stemming does not guarantee a dictionary word; lemmatization aims to return a linguistically meaningful base form.

---

## 5. Linguistic Analysis

Includes:

- Penn Treebank POS tags
- Universal POS tags
- dependency parsing
- constituency parsing demonstration
- NER with BIO tags
- simple coreference demonstration

---

## 6. Semantics

Includes:

- WordNet relations
- Lesk Word Sense Disambiguation
- embedding-based WSD
- semantic role demonstration

---

## 7. Vectorization

Implements:

- Bag of Words
- TF-IDF from scratch
- validation against a library-based calculation
- BM25
- top TF-IDF terms per emotion class

Scratch TF-IDF validation passed with only tiny floating-point differences.

---

## 8. N-gram Language Modelling

Implements unigram, bigram and trigram models from scratch.

Methods:

- Laplace smoothing
- absolute-discount backoff
- held-out perplexity
- NLTK validation
- sentence generation

Final backoff perplexities:

| Model | Backoff Perplexity |
|---|---:|
| Unigram | 548.5972 |
| Bigram | 216.9287 |
| Trigram | 196.7952 |

The trigram backoff model produced the lowest perplexity.

---

## 9. Classical Machine Learning

Models:

- Multinomial Naive Bayes from scratch
- Logistic Regression
- Linear SVM

Features:

- TF-IDF
- up to 10,000 features

Final test results:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Naive Bayes from scratch | 0.7820 | 0.6082 |
| Logistic Regression | 0.8880 | 0.8302 |
| Linear SVM | 0.9045 | 0.8589 |

**Best classical model:** Linear SVM.

---

## 10. Sequence Labelling

### HMM + Viterbi POS Tagging

Implemented from scratch using the Brown corpus with Universal POS tags.

Result:

```text
HMM POS Accuracy = 0.8590
```

### CRF Named Entity Recognition

Uses CoNLL-2003 with hand-engineered CRF features:

- current word
- lowercase word
- suffixes
- capitalization
- digit status
- word length
- previous word context
- next word context
- BOS/EOS flags

Results:

```text
Token Macro-F1 = 0.7231
Strict Entity-Level F1 = 0.7871
```

---

## 11. Topic Modelling

Compares:

- LSA
- LDA
- BERTopic

Final coherence comparison:

| Method | Coherence |
|---|---:|
| LSA | 0.0871 |
| LDA | 0.0538 |
| BERTopic | 0.1502 |

BERTopic produced the most interpretable clusters, including themes related to:

- fear/anxiety
- anger
- relationships
- family
- loneliness
- books
- music
- sleep/energy
- joy

---

## 12. Word Embeddings

Includes:

- Skip-gram with Negative Sampling from scratch
- gensim CBOW
- gensim Skip-gram
- pretrained GloVe
- FastText
- contextual BERT embeddings

Scratch Word2Vec training loss decreased across epochs:

```text
1.3027 -> 1.1236 -> 1.0038
```

### Analogy

Using pretrained GloVe:

```text
king - man + woman ~= queen
```

`queen` was ranked first.

### FastText OOV Demonstration

An unseen word such as:

```text
happytastic
```

was not in the exact training vocabulary, but FastText still generated a vector using subword information.

### Contextual BERT Demonstration

The word `bank` was embedded in:

```text
I deposited money in the bank.
I sat beside the river bank.
```

The contextual vectors differed:

```text
Cosine similarity = 0.5118
```

### Embeddings + Classifier

Average pretrained GloVe vectors + Logistic Regression:

```text
Accuracy = 0.4090
Macro-F1 = 0.3684
```

---

## 13. Transformer Fine-Tuning

Model:

```text
distilbert-base-uncased
```

The model was fine-tuned on the full dataset:

```text
Train      = 16,000
Validation = 2,000
Test       = 2,000
```

Class weighting was used to improve minority-class performance.

Best validation checkpoint was selected by Macro-F1.

Final test results:

| Metric | Score |
|---|---:|
| Accuracy | 0.9200 |
| Macro Precision | 0.8622 |
| Macro Recall | 0.9253 |
| Macro-F1 | 0.8862 |
| Weighted F1 | 0.9224 |

Comparison:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Linear SVM | 0.9045 | 0.8589 |
| **DistilBERT** | **0.9200** | **0.8862** |

DistilBERT improved Macro-F1 over the strongest classical baseline by:

```text
+0.0273
```

---

## 14. Retrieval-Augmented Generation

The RAG demonstration uses a small secondary knowledge base.

### Hybrid Retrieval

Sparse retrieval:

```text
TF-IDF
```

Dense retrieval:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Generator:

```text
google/flan-t5-small
```

Hybrid weights:

```text
Sparse = 0.5
Dense  = 0.5
```

### Successful Query

```text
What is Retrieval-Augmented Generation and how does it work?
```

Top retrieval:

```text
RAG
Sparse score = 0.7002
Dense score  = 0.7514
```

Generated answer:

```text
combines information retrieval with text generation
```

### Out-of-Knowledge Query

```text
Who won the FIFA World Cup in 2022?
```

The knowledge base did not contain the answer.

The retrieval confidence gate detected weak retrieval and blocked generation:

```text
I do not have enough information.
```

This reduces unsupported generation and hallucination.

---

## 15. Final Evaluation

Final classification comparison:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Naive Bayes from scratch | 0.7820 | 0.6082 |
| Logistic Regression | 0.8880 | 0.8302 |
| Linear SVM | 0.9045 | 0.8589 |
| GloVe + Logistic Regression | 0.4090 | 0.3684 |
| **DistilBERT** | **0.9200** | **0.8862** |

### Final Conclusion

DistilBERT produced the strongest emotion-classification result because contextual Transformer representations capture semantic relationships that static feature representations cannot model as effectively.

Linear SVM remains an excellent efficiency/performance baseline because it achieves strong Macro-F1 with much lower computational cost.

The project also demonstrates that:

- TF-IDF can outperform simple averaged static embeddings on emotion classification.
- Backoff substantially improves higher-order n-gram language models.
- CRFs provide effective sequence labelling with interpretable hand-engineered features.
- BERTopic produces more semantically meaningful topic clusters than LSA/LDA for this corpus.
- contextual BERT embeddings change with word meaning and sentence context.
- hybrid RAG retrieval can support grounded generation.
- retrieval-confidence thresholds can reduce hallucination on unsupported questions.

---

# Running the Project

The complete pipeline can currently be launched with:

```powershell
python main.py
```

Individual modules can also be executed directly, for example:

```powershell
python -c "import config; from src import rag; rag.run(config)"
```

or:

```powershell
python -c "import config; from src import evaluation; evaluation.run(config)"
```

For long-running stages such as Transformer fine-tuning, it is recommended to run only that stage instead of re-running the entire pipeline.

---

# Important Output Files

```text
outputs/
├── classical_ml/
│   ├── classical_ml_report.txt
│   ├── classical_ml_results.json
│   └── *_confusion_matrix.png
│
├── sequence_labeling/
│   └── sequence_labeling_report.txt
│
├── topic_modeling/
│   ├── topic_modeling_report.txt
│   └── bertopic_model_improved/
│
├── embeddings/
│   ├── embeddings_report.txt
│   ├── scratch_word2vec.pt
│   ├── gensim_cbow.model
│   ├── gensim_skipgram.model
│   └── fasttext.model
│
├── transformer/
│   ├── transformer_report.txt
│   ├── transformer_results.json
│   ├── distilbert_confusion_matrix.png
│   └── best_distilbert/
│
├── rag/
│   ├── rag_report.txt
│   └── rag_results.json
│
└── final_evaluation/
    ├── final_evaluation_report.txt
    ├── final_evaluation_results.json
    └── classification_macro_f1_comparison.png
```

---

# Reproducibility

Random seeds are controlled through `config.py`.

The project uses:

```text
seed = 42
```

where supported.

---

# Notes

- The project was developed and tested using Python 3.11.
- Hugging Face models are downloaded automatically on first use.
- Transformer training on CPU is significantly slower than classical ML.
- Saved models and outputs allow expensive experiments to be inspected without retraining.
- Some pretrained model files are large and may be better excluded from GitHub using `.gitignore`.

---

# Author

Shilpa Golla
