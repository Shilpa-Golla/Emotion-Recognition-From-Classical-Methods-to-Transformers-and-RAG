"""
Emotion demo page.

Type a sentence and see the predicted emotion from:
  - Linear SVM on TF-IDF (retrained here in a few seconds, same settings as Stage 9)
  - Fine-tuned DistilBERT (only if the trained weights exist on this computer)

Run:  python app.py      then open the link it prints (http://127.0.0.1:7860)
"""

import glob
import os

import gradio as gr
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import config

LABELS = config.LABEL_NAMES


# ---------------------------------------------------------
# 1. Linear SVM (retrained at start-up, takes a few seconds)
# ---------------------------------------------------------
def train_svm():
    print("Loading dataset and training Linear SVM ...")
    train = load_dataset(config.DATASET_NAME)["train"]
    vectorizer = TfidfVectorizer(
        max_features=config.CLASSICAL_MAX_FEATURES,
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(train["text"])
    model = LinearSVC(C=config.SVM_C, random_state=config.SEED)
    model.fit(X, train["label"])
    print("Linear SVM ready.")
    return vectorizer, model


# ---------------------------------------------------------
# 2. DistilBERT (loaded only if trained weights are found)
# ---------------------------------------------------------
def find_distilbert_folder():
    """Look for a best_distilbert folder that contains real weights."""
    for folder in glob.glob("**/best_distilbert", recursive=True):
        has_weights = (
            os.path.exists(os.path.join(folder, "model.safetensors"))
            or os.path.exists(os.path.join(folder, "pytorch_model.bin"))
        )
        if has_weights:
            return folder
    return None


def load_distilbert():
    folder = find_distilbert_folder()
    if folder is None:
        print("DistilBERT weights not found: only the SVM will be shown.")
        return None, None
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading DistilBERT from {folder} ...")
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder)
    model.eval()
    return tokenizer, model


vectorizer, svm = train_svm()
bert_tokenizer, bert_model = load_distilbert()


# ---------------------------------------------------------
# 3. Prediction
# ---------------------------------------------------------
def predict(sentence):
    sentence = (sentence or "").strip().lower()
    if not sentence:
        return "Type a sentence first.", {}

    svm_label = LABELS[int(svm.predict(vectorizer.transform([sentence]))[0])]

    if bert_model is None:
        return svm_label, {"DistilBERT weights not found": 1.0}

    import torch

    inputs = bert_tokenizer(
        sentence,
        truncation=True,
        max_length=config.TRANSFORMER_MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = bert_model(**inputs).logits[0].numpy()
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return svm_label, {LABELS[i]: float(p) for i, p in enumerate(probs)}


# ---------------------------------------------------------
# 4. Page
# ---------------------------------------------------------
EXAMPLES = [
    "i am not happy at all",
    "i cant believe i actually won the lottery",
    "i miss my grandmother so much",
    "i feel so nervous about my exam tomorrow",
    "i am furious that they cancelled my flight again",
    "i feel so loved when you hold my hand",
]

with gr.Blocks(title="Emotion Recognition") as page:
    gr.Markdown(
        "# Emotion Recognition\n"
        "Type an English sentence and compare the classical baseline "
        "(TF-IDF + Linear SVM) with the fine-tuned DistilBERT."
    )
    text_box = gr.Textbox(label="Your sentence", placeholder="i feel ...", lines=2)
    predict_button = gr.Button("Predict emotion", variant="primary")
    with gr.Row():
        svm_output = gr.Textbox(label="Linear SVM prediction")
        bert_output = gr.Label(label="DistilBERT probabilities", num_top_classes=6)
    gr.Examples(examples=EXAMPLES, inputs=text_box)

    predict_button.click(predict, inputs=text_box, outputs=[svm_output, bert_output])
    text_box.submit(predict, inputs=text_box, outputs=[svm_output, bert_output])


if __name__ == "__main__":
    page.launch()