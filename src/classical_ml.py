import os
import json

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import config


# =========================================================
# MULTINOMIAL NAIVE BAYES FROM SCRATCH
# =========================================================

class MultinomialNBFromScratch:

    def __init__(self, alpha=1.0):

        self.alpha = alpha

        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    def fit(self, X, y):

        y = np.asarray(y)

        self.classes_ = np.unique(y)

        number_of_classes = len(
            self.classes_
        )

        number_of_features = X.shape[1]

        self.class_log_prior_ = np.zeros(
            number_of_classes
        )

        self.feature_log_prob_ = np.zeros(
            (
                number_of_classes,
                number_of_features
            )
        )

        number_of_samples = len(y)

        for class_index, class_label in enumerate(
            self.classes_
        ):

            # Documents belonging to this class
            class_mask = (
                y == class_label
            )

            X_class = X[
                class_mask
            ]

            number_in_class = (
                class_mask.sum()
            )

            # --------------------------------
            # CLASS PRIOR
            # P(class)
            # --------------------------------

            self.class_log_prior_[
                class_index
            ] = np.log(
                number_in_class
                / number_of_samples
            )

            # --------------------------------
            # FEATURE COUNTS
            # --------------------------------

            feature_counts = np.asarray(
                X_class.sum(
                    axis=0
                )
            ).ravel()

            # Laplace smoothing
            smoothed_counts = (
                feature_counts
                + self.alpha
            )

            total_smoothed_count = (
                smoothed_counts.sum()
            )

            # --------------------------------
            # P(word | class)
            # --------------------------------

            self.feature_log_prob_[
                class_index
            ] = np.log(
                smoothed_counts
                / total_smoothed_count
            )

        return self


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    def predict(self, X):

        # log P(class)
        # +
        # sum TF-IDF(word) *
        # log P(word | class)

        scores = (
            X
            @ self.feature_log_prob_.T
        )

        scores = (
            np.asarray(scores)
            +
            self.class_log_prior_
        )

        predicted_indices = np.argmax(
            scores,
            axis=1
        )

        return self.classes_[
            predicted_indices
        ]


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    label_names,
    output_dir
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

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        model_name
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy:        "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{recall:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{f1:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0
    )

    print(
        report
    )

    # --------------------------------
    # CONFUSION MATRIX
    # --------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved confusion matrix:",
        path
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            precision
        ),
        "macro_recall": float(
            recall
        ),
        "macro_f1": float(
            f1
        )
    }


# =========================================================
# RUN
# =========================================================

def run(config_module=config):

    print(
        "\nStarting Classical ML Stage..."
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

    train_labels = dataset[
        "train"
    ][
        config_module.LABEL_COLUMN
    ]

    validation_texts = dataset[
        "validation"
    ][
        config_module.TEXT_COLUMN
    ]

    validation_labels = dataset[
        "validation"
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
        "\nTrain:",
        len(train_texts)
    )

    print(
        "Validation:",
        len(validation_texts)
    )

    print(
        "Test:",
        len(test_texts)
    )


    # =====================================================
    # TF-IDF
    # =====================================================

    print(
        "\n===== TF-IDF FEATURES ====="
    )

    vectorizer = TfidfVectorizer(
        max_features=(
            config_module
            .CLASSICAL_MAX_FEATURES
        ),
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train = (
        vectorizer.fit_transform(
            train_texts
        )
    )

    X_validation = (
        vectorizer.transform(
            validation_texts
        )
    )

    X_test = (
        vectorizer.transform(
            test_texts
        )
    )

    print(
        "Training matrix:",
        X_train.shape
    )

    print(
        "Validation matrix:",
        X_validation.shape
    )

    print(
        "Test matrix:",
        X_test.shape
    )


    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================

    output_dir = os.path.join(
        config_module.OUTPUT_DIR,
        "classical_ml"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # =====================================================
    # MODEL 1:
    # NAIVE BAYES FROM SCRATCH
    # =====================================================

    print(
        "\n===== TRAINING NAIVE BAYES "
        "FROM SCRATCH ====="
    )

    nb_model = MultinomialNBFromScratch(
        alpha=config_module.NB_ALPHA
    )

    nb_model.fit(
        X_train,
        train_labels
    )

    nb_validation_predictions = (
        nb_model.predict(
            X_validation
        )
    )

    nb_validation_f1 = f1_score(
        validation_labels,
        nb_validation_predictions,
        average="macro",
        zero_division=0
    )

    print(
        "Validation Macro-F1:",
        round(
            nb_validation_f1,
            4
        )
    )

    nb_test_predictions = (
        nb_model.predict(
            X_test
        )
    )

    nb_results = evaluate_model(
        "Naive Bayes From Scratch",
        test_labels,
        nb_test_predictions,
        config_module.LABEL_NAMES,
        output_dir
    )


    # =====================================================
    # MODEL 2:
    # LOGISTIC REGRESSION
    # =====================================================

    print(
        "\n===== TRAINING "
        "LOGISTIC REGRESSION ====="
    )

    logistic_model = (
        LogisticRegression(
            max_iter=(
                config_module
                .LOGISTIC_MAX_ITER
            ),
            random_state=(
                config_module.SEED
            )
        )
    )

    logistic_model.fit(
        X_train,
        train_labels
    )

    logistic_validation_predictions = (
        logistic_model.predict(
            X_validation
        )
    )

    logistic_validation_f1 = f1_score(
        validation_labels,
        logistic_validation_predictions,
        average="macro",
        zero_division=0
    )

    print(
        "Validation Macro-F1:",
        round(
            logistic_validation_f1,
            4
        )
    )

    logistic_test_predictions = (
        logistic_model.predict(
            X_test
        )
    )

    logistic_results = evaluate_model(
        "Logistic Regression",
        test_labels,
        logistic_test_predictions,
        config_module.LABEL_NAMES,
        output_dir
    )


    # =====================================================
    # MODEL 3:
    # LINEAR SVM
    # =====================================================

    print(
        "\n===== TRAINING "
        "LINEAR SVM ====="
    )

    svm_model = LinearSVC(
        C=config_module.SVM_C,
        random_state=config_module.SEED
    )

    svm_model.fit(
        X_train,
        train_labels
    )

    svm_validation_predictions = (
        svm_model.predict(
            X_validation
        )
    )

    svm_validation_f1 = f1_score(
        validation_labels,
        svm_validation_predictions,
        average="macro",
        zero_division=0
    )

    print(
        "Validation Macro-F1:",
        round(
            svm_validation_f1,
            4
        )
    )

    svm_test_predictions = (
        svm_model.predict(
            X_test
        )
    )

    svm_results = evaluate_model(
        "Linear SVM",
        test_labels,
        svm_test_predictions,
        config_module.LABEL_NAMES,
        output_dir
    )


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    results = {
        "Naive Bayes From Scratch":
            nb_results,

        "Logistic Regression":
            logistic_results,

        "Linear SVM":
            svm_results
    }

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CLASSICAL ML TEST RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        f"{'Model':<28}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'Macro-F1':<12}"
    )

    print(
        "-" * 75
    )

    for model_name, metrics in (
        results.items()
    ):

        print(
            f"{model_name:<28}"
            f"{metrics['accuracy']:<12.4f}"
            f"{metrics['macro_precision']:<12.4f}"
            f"{metrics['macro_recall']:<12.4f}"
            f"{metrics['macro_f1']:<12.4f}"
        )


    # =====================================================
    # BEST CLASSICAL MODEL
    # =====================================================

    best_model = max(
        results,
        key=lambda name:
            results[name]["macro_f1"]
    )

    print(
        "\nBest classical model:",
        best_model
    )

    print(
        "Best Macro-F1:",
        round(
            results[
                best_model
            ]["macro_f1"],
            4
        )
    )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    json_path = os.path.join(
        output_dir,
        "classical_ml_results.json"
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

    report_path = os.path.join(
        output_dir,
        "classical_ml_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CLASSICAL MACHINE LEARNING "
            "REPORT\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "Features: TF-IDF\n"
        )

        file.write(
            "Models:\n"
        )

        file.write(
            "1. Multinomial Naive Bayes "
            "from scratch\n"
        )

        file.write(
            "2. Logistic Regression\n"
        )

        file.write(
            "3. Linear SVM\n\n"
        )

        for model_name, metrics in (
            results.items()
        ):

            file.write(
                f"{model_name}\n"
            )

            file.write(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}\n"
            )

            file.write(
                "Macro Precision: "
                f"{metrics['macro_precision']:.4f}\n"
            )

            file.write(
                f"Macro Recall: "
                f"{metrics['macro_recall']:.4f}\n"
            )

            file.write(
                f"Macro F1: "
                f"{metrics['macro_f1']:.4f}\n\n"
            )

        file.write(
            f"Best model: "
            f"{best_model}\n"
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
        "\nClassical ML Stage "
        "Completed Successfully!"
    )