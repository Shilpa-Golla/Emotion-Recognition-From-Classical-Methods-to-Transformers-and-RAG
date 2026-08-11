import json
import os
import random
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
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
        torch.cuda.manual_seed_all(seed)


# =========================================================
# TOKENIZE DATASET
# =========================================================

def tokenize_dataset(
    dataset,
    tokenizer,
    config_module
):

    def tokenize_batch(batch):

        return tokenizer(
            batch[
                config_module.TEXT_COLUMN
            ],
            truncation=True,
            max_length=(
                config_module
                .TRANSFORMER_MAX_LENGTH
            )
        )

    tokenized_dataset = (
        dataset.map(
            tokenize_batch,
            batched=True
        )
    )

    return tokenized_dataset


# =========================================================
# CREATE DATA LOADER
# =========================================================

def create_dataloader(
    dataset,
    tokenizer,
    batch_size,
    shuffle
):

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors="pt"
        )
    )

    dataset = dataset.rename_column(
        "label",
        "labels"
    )

    keep_columns = [
        "input_ids",
        "attention_mask",
        "labels"
    ]

    remove_columns = [
        column
        for column
        in dataset.column_names
        if column not in keep_columns
    ]

    dataset = dataset.remove_columns(
        remove_columns
    )

    dataset.set_format(
        type="torch"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )

    return loader


# =========================================================
# CLASS WEIGHTS
# =========================================================

def calculate_class_weights(
    labels,
    number_of_classes
):

    label_counts = np.bincount(
        labels,
        minlength=number_of_classes
    )

    total = len(labels)

    weights = (
        total
        /
        (
            number_of_classes
            *
            label_counts
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_function,
    device
):

    model.train()

    total_loss = 0.0

    predictions = []
    true_labels = []

    for batch_number, batch in enumerate(
        dataloader,
        start=1
    ):

        input_ids = (
            batch["input_ids"]
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(device)
        )

        labels = (
            batch["labels"]
            .to(device)
        )

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        loss = loss_function(
            logits,
            labels
        )

        loss.backward()

        # Helps avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        batch_predictions = (
            torch.argmax(
                logits,
                dim=1
            )
        )

        predictions.extend(
            batch_predictions
            .detach()
            .cpu()
            .numpy()
        )

        true_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
        )

        if batch_number % 50 == 0:

            print(
                f"  Batch "
                f"{batch_number}/"
                f"{len(dataloader)}"
            )

    average_loss = (
        total_loss
        /
        max(
            len(dataloader),
            1
        )
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        macro_f1
    )


# =========================================================
# EVALUATE
# =========================================================

def evaluate_model(
    model,
    dataloader,
    device
):

    model.eval()

    predictions = []
    true_labels = []

    total_loss = 0.0

    loss_function = (
        torch.nn.CrossEntropyLoss()
    )

    with torch.no_grad():

        for batch in dataloader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            loss = loss_function(
                logits,
                labels
            )

            total_loss += (
                loss.item()
            )

            batch_predictions = (
                torch.argmax(
                    logits,
                    dim=1
                )
            )

            predictions.extend(
                batch_predictions
                .detach()
                .cpu()
                .numpy()
            )

            true_labels.extend(
                labels
                .detach()
                .cpu()
                .numpy()
            )

    average_loss = (
        total_loss
        /
        max(
            len(dataloader),
            1
        )
    )

    return (
        np.array(true_labels),
        np.array(predictions),
        average_loss
    )


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy":
            float(accuracy),

        "macro_precision":
            float(precision),

        "macro_recall":
            float(recall),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1)
    }


# =========================================================
# LOAD CLASSICAL BASELINE
# =========================================================

def load_classical_baseline(
    config_module
):

    path = os.path.join(
        config_module.OUTPUT_DIR,
        "classical_ml",
        "classical_ml_results.json"
    )

    if not os.path.exists(path):

        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        results = json.load(file)

    best_model = max(
        results,
        key=lambda model_name:
            results[
                model_name
            ]["macro_f1"]
    )

    return {
        "model":
            best_model,

        "accuracy":
            results[
                best_model
            ]["accuracy"],

        "macro_f1":
            results[
                best_model
            ]["macro_f1"]
    }


# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    label_names,
    output_dir
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_names
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        "DistilBERT Confusion Matrix"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "distilbert_confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return path


# =========================================================
# DEMONSTRATE ATTENTION CONCEPT
# =========================================================

def attention_explanation():

    explanation = """
TRANSFORMER ATTENTION SUMMARY

Scaled Dot-Product Attention:

Attention(Q, K, V)
    = softmax(QK^T / sqrt(d_k)) V

Q = Query
K = Key
V = Value

Why divide by sqrt(d_k)?
It prevents very large dot-product values from
making softmax extremely peaked.

Multi-Head Attention:
Several attention heads learn different relationships
between words in parallel.

Unlike older sequential RNN/LSTM processing,
Transformer self-attention can directly model
relationships between tokens across the sequence.
"""

    return explanation


# =========================================================
# RUN
# =========================================================

def run(
    config_module=config
):

    print(
        "\nStarting Transformer / "
        "DistilBERT Stage..."
    )

    set_all_seeds(
        config_module.SEED
    )

    start_time = time.time()


    # =====================================================
    # DEVICE
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nTraining device:",
        device
    )

    if device.type == "cpu":

        print(
            "CPU detected. "
            "Transformer training "
            "will be slower."
        )


    # =====================================================
    # LOAD DATASET
    # =====================================================

    dataset = load_dataset(
        config_module.DATASET_NAME
    )

    train_dataset = (
        dataset["train"]
    )

    validation_dataset = (
        dataset["validation"]
    )

    test_dataset = (
        dataset["test"]
    )


    # =====================================================
    # FAST MODE
    # =====================================================

    if (
        config_module
        .TRANSFORMER_FAST_MODE
    ):

        print(
            "\nFAST MODE ENABLED"
        )

        print(
            "This run checks that "
            "the Transformer pipeline "
            "works before final training."
        )

        train_size = min(
            config_module
            .TRANSFORMER_FAST_TRAIN_SIZE,
            len(train_dataset)
        )

        validation_size = min(
            config_module
            .TRANSFORMER_FAST_VAL_SIZE,
            len(validation_dataset)
        )

        test_size = min(
            config_module
            .TRANSFORMER_FAST_TEST_SIZE,
            len(test_dataset)
        )

        train_dataset = (
            train_dataset
            .shuffle(
                seed=config_module.SEED
            )
            .select(
                range(train_size)
            )
        )

        validation_dataset = (
            validation_dataset
            .shuffle(
                seed=config_module.SEED
            )
            .select(
                range(validation_size)
            )
        )

        test_dataset = (
            test_dataset
            .shuffle(
                seed=config_module.SEED
            )
            .select(
                range(test_size)
            )
        )

    else:

        print(
            "\nFINAL MODE ENABLED"
        )

        print(
            "Using complete "
            "train/validation/test splits."
        )


    print(
        "\nTraining examples:",
        len(train_dataset)
    )

    print(
        "Validation examples:",
        len(validation_dataset)
    )

    print(
        "Test examples:",
        len(test_dataset)
    )


    # =====================================================
    # TOKENIZER
    # =====================================================

    print(
        "\nLoading tokenizer:"
    )

    print(
        config_module
        .TRANSFORMER_MODEL
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            config_module
            .TRANSFORMER_MODEL
        )
    )


    # =====================================================
    # TOKENIZE
    # =====================================================

    print(
        "\nTokenizing dataset..."
    )

    train_tokenized = (
        tokenize_dataset(
            train_dataset,
            tokenizer,
            config_module
        )
    )

    validation_tokenized = (
        tokenize_dataset(
            validation_dataset,
            tokenizer,
            config_module
        )
    )

    test_tokenized = (
        tokenize_dataset(
            test_dataset,
            tokenizer,
            config_module
        )
    )


    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader = (
        create_dataloader(
            train_tokenized,
            tokenizer,
            config_module
            .TRANSFORMER_BATCH_SIZE,
            shuffle=True
        )
    )

    validation_loader = (
        create_dataloader(
            validation_tokenized,
            tokenizer,
            config_module
            .TRANSFORMER_BATCH_SIZE,
            shuffle=False
        )
    )

    test_loader = (
        create_dataloader(
            test_tokenized,
            tokenizer,
            config_module
            .TRANSFORMER_BATCH_SIZE,
            shuffle=False
        )
    )


    # =====================================================
    # LABEL MAPPINGS
    # =====================================================

    id2label = {
        index: label
        for index, label
        in enumerate(
            config_module
            .LABEL_NAMES
        )
    }

    label2id = {
        label: index
        for index, label
        in id2label.items()
    }


    # =====================================================
    # LOAD MODEL
    # =====================================================

    print(
        "\nLoading DistilBERT "
        "classification model..."
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            config_module
            .TRANSFORMER_MODEL,
            num_labels=len(
                config_module
                .LABEL_NAMES
            ),
            id2label=id2label,
            label2id=label2id
        )
    )

    model = model.to(
        device
    )


    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    training_labels = (
        train_dataset[
            config_module
            .LABEL_COLUMN
        ]
    )

    class_weights = (
        calculate_class_weights(
            training_labels,
            len(
                config_module
                .LABEL_NAMES
            )
        )
        .to(device)
    )

    print(
        "\nClass weights:"
    )

    for (
        label_name,
        weight
    ) in zip(
        config_module.LABEL_NAMES,
        class_weights
        .detach()
        .cpu()
        .numpy()
    ):

        print(
            f"{label_name:<12}"
            f"{weight:.4f}"
        )

    loss_function = (
        torch.nn.CrossEntropyLoss(
            weight=class_weights
        )
    )


    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=(
                config_module
                .TRANSFORMER_LEARNING_RATE
            )
        )
    )


    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================

    output_dir = os.path.join(
        config_module.OUTPUT_DIR,
        "transformer"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # =====================================================
    # TRAINING
    # =====================================================

    best_validation_f1 = -1.0

    best_model_path = os.path.join(
        output_dir,
        "best_distilbert"
    )

    training_history = []

    print(
        "\n"
        + "=" * 60
    )

    print(
        "STARTING DISTILBERT "
        "FINE-TUNING"
    )

    print(
        "=" * 60
    )

    for epoch in range(
        config_module
        .TRANSFORMER_EPOCHS
    ):

        print(
            f"\nEpoch "
            f"{epoch + 1}/"
            f"{config_module.TRANSFORMER_EPOCHS}"
        )

        epoch_start = (
            time.time()
        )

        (
            train_loss,
            train_macro_f1
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device
        )

        (
            validation_true,
            validation_predictions,
            validation_loss
        ) = evaluate_model(
            model,
            validation_loader,
            device
        )

        validation_metrics = (
            calculate_metrics(
                validation_true,
                validation_predictions
            )
        )

        epoch_time = (
            time.time()
            -
            epoch_start
        )

        print(
            "\nTrain Loss:",
            round(
                train_loss,
                4
            )
        )

        print(
            "Train Macro-F1:",
            round(
                train_macro_f1,
                4
            )
        )

        print(
            "Validation Loss:",
            round(
                validation_loss,
                4
            )
        )

        print(
            "Validation Accuracy:",
            round(
                validation_metrics[
                    "accuracy"
                ],
                4
            )
        )

        print(
            "Validation Macro-F1:",
            round(
                validation_metrics[
                    "macro_f1"
                ],
                4
            )
        )

        print(
            "Epoch time:",
            round(
                epoch_time,
                2
            ),
            "seconds"
        )

        training_history.append(
            {
                "epoch":
                    epoch + 1,

                "train_loss":
                    float(
                        train_loss
                    ),

                "train_macro_f1":
                    float(
                        train_macro_f1
                    ),

                "validation_loss":
                    float(
                        validation_loss
                    ),

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_macro_f1":
                    validation_metrics[
                        "macro_f1"
                    ]
            }
        )

        # --------------------------------
        # SAVE BEST MODEL
        # --------------------------------

        if (
            validation_metrics[
                "macro_f1"
            ]
            >
            best_validation_f1
        ):

            best_validation_f1 = (
                validation_metrics[
                    "macro_f1"
                ]
            )

            model.save_pretrained(
                best_model_path
            )

            tokenizer.save_pretrained(
                best_model_path
            )

            print(
                "Best model updated."
            )


    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    print(
        "\nLoading best validation model..."
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            best_model_path
        )
        .to(device)
    )


    # =====================================================
    # FINAL TEST
    # =====================================================

    (
        test_true,
        test_predictions,
        test_loss
    ) = evaluate_model(
        model,
        test_loader,
        device
    )

    test_metrics = (
        calculate_metrics(
            test_true,
            test_predictions
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "DISTILBERT FINAL TEST RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy:        "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:     "
        f"{test_metrics['weighted_f1']:.4f}"
    )


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    test_report = (
        classification_report(
            test_true,
            test_predictions,
            target_names=(
                config_module
                .LABEL_NAMES
            ),
            digits=4,
            zero_division=0
        )
    )

    print(
        "\nClassification Report:"
    )

    print(
        test_report
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    confusion_path = (
        save_confusion_matrix(
            test_true,
            test_predictions,
            config_module
            .LABEL_NAMES,
            output_dir
        )
    )

    print(
        "Saved confusion matrix:",
        confusion_path
    )


    # =====================================================
    # COMPARE AGAINST SVM
    # =====================================================

    classical_baseline = (
        load_classical_baseline(
            config_module
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TRANSFORMER VS "
        "CLASSICAL BASELINE"
    )

    print(
        "=" * 60
    )

    if classical_baseline:

        print(
            f"{'Model':<30}"
            f"{'Accuracy':<15}"
            f"{'Macro-F1':<15}"
        )

        print(
            "-" * 60
        )

        print(
            f"{classical_baseline['model']:<30}"
            f"{classical_baseline['accuracy']:<15.4f}"
            f"{classical_baseline['macro_f1']:<15.4f}"
        )

        print(
            f"{'DistilBERT':<30}"
            f"{test_metrics['accuracy']:<15.4f}"
            f"{test_metrics['macro_f1']:<15.4f}"
        )

        difference = (
            test_metrics[
                "macro_f1"
            ]
            -
            classical_baseline[
                "macro_f1"
            ]
        )

        print(
            "\nMacro-F1 difference "
            "vs classical baseline:",
            f"{difference:+.4f}"
        )

        if difference > 0:

            print(
                "DistilBERT beats "
                "the classical baseline."
            )

        else:

            print(
                "The classical baseline "
                "remains stronger in "
                "this experiment."
            )

    else:

        print(
            "Classical baseline JSON "
            "was not found."
        )


    # =====================================================
    # ATTENTION EXPLANATION
    # =====================================================

    attention_text = (
        attention_explanation()
    )

    print(
        "\n"
        + attention_text
    )


    # =====================================================
    # TRAINING TIME
    # =====================================================

    total_training_time = (
        time.time()
        -
        start_time
    )

    print(
        "\nTotal Transformer "
        "stage time:",
        round(
            total_training_time,
            2
        ),
        "seconds"
    )


    # =====================================================
    # SAVE JSON RESULTS
    # =====================================================

    results = {
        "model":
            config_module
            .TRANSFORMER_MODEL,

        "fast_mode":
            config_module
            .TRANSFORMER_FAST_MODE,

        "train_examples":
            len(train_dataset),

        "validation_examples":
            len(
                validation_dataset
            ),

        "test_examples":
            len(test_dataset),

        "epochs":
            config_module
            .TRANSFORMER_EPOCHS,

        "best_validation_macro_f1":
            float(
                best_validation_f1
            ),

        "test_metrics":
            test_metrics,

        "training_history":
            training_history,

        "total_time_seconds":
            float(
                total_training_time
            )
    }

    json_path = os.path.join(
        output_dir,
        "transformer_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )


    # =====================================================
    # SAVE TEXT REPORT
    # =====================================================

    report_path = os.path.join(
        output_dir,
        "transformer_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "TRANSFORMER / "
            "DISTILBERT REPORT\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Model: "
            f"{config_module.TRANSFORMER_MODEL}\n"
        )

        file.write(
            f"Fast mode: "
            f"{config_module.TRANSFORMER_FAST_MODE}\n"
        )

        file.write(
            f"Train examples: "
            f"{len(train_dataset)}\n"
        )

        file.write(
            f"Validation examples: "
            f"{len(validation_dataset)}\n"
        )

        file.write(
            f"Test examples: "
            f"{len(test_dataset)}\n"
        )

        file.write(
            f"Epochs: "
            f"{config_module.TRANSFORMER_EPOCHS}\n\n"
        )

        file.write(
            "FINAL TEST METRICS\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        for (
            metric_name,
            value
        ) in test_metrics.items():

            file.write(
                f"{metric_name}: "
                f"{value:.4f}\n"
            )

        file.write(
            "\nCLASSIFICATION REPORT\n"
        )

        file.write(
            test_report
        )

        if classical_baseline:

            file.write(
                "\n\nCLASSICAL "
                "BASELINE COMPARISON\n"
            )

            file.write(
                f"Best baseline: "
                f"{classical_baseline['model']}\n"
            )

            file.write(
                "Baseline Accuracy: "
                f"{classical_baseline['accuracy']:.4f}\n"
            )

            file.write(
                "Baseline Macro-F1: "
                f"{classical_baseline['macro_f1']:.4f}\n"
            )

            file.write(
                "DistilBERT Accuracy: "
                f"{test_metrics['accuracy']:.4f}\n"
            )

            file.write(
                "DistilBERT Macro-F1: "
                f"{test_metrics['macro_f1']:.4f}\n"
            )

        file.write(
            "\n"
            + attention_text
        )

        file.write(
            "\nTotal stage time: "
            f"{total_training_time:.2f} seconds\n"
        )

    print(
        "\nSaved:",
        report_path
    )

    print(
        "Saved:",
        json_path
    )

    print(
        "Saved best model:",
        best_model_path
    )

    print(
        "\nTransformer / "
        "DistilBERT Stage "
        "Completed Successfully!"
    )

    return test_metrics