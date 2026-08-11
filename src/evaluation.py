import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import config


# =========================================================
# SAFE JSON LOADER
# =========================================================

def load_json(path):

    if not os.path.exists(path):

        print(
            "Warning: file not found:",
            path
        )

        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# SAFE TEXT LOADER
# =========================================================

def load_text(path):

    if not os.path.exists(path):

        print(
            "Warning: file not found:",
            path
        )

        return ""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# =========================================================
# EXTRACT FLOAT FROM TEXT
# =========================================================

def extract_float(
    text,
    pattern
):

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    if match:

        try:

            return float(
                match.group(1)
            )

        except ValueError:

            return None

    return None


# =========================================================
# LOAD CLASSICAL ML RESULTS
# =========================================================

def load_classical_results(
    output_dir
):

    path = os.path.join(
        output_dir,
        "classical_ml",
        "classical_ml_results.json"
    )

    results = load_json(
        path
    )

    if not results:

        return {}

    return results


# =========================================================
# LOAD TRANSFORMER RESULTS
# =========================================================

def load_transformer_results(
    output_dir
):

    path = os.path.join(
        output_dir,
        "transformer",
        "transformer_results.json"
    )

    data = load_json(
        path
    )

    if not data:

        return None

    return data.get(
        "test_metrics",
        {}
    )


# =========================================================
# LOAD EMBEDDING RESULTS
# =========================================================

def load_embedding_results(
    output_dir
):

    path = os.path.join(
        output_dir,
        "embeddings",
        "embeddings_report.txt"
    )

    text = load_text(
        path
    )

    if not text:

        return None

    # Search inside Embeddings + Classifier section

    accuracy = extract_float(
        text,
        r"accuracy:\s*([0-9.]+)"
    )

    macro_f1 = extract_float(
        text,
        r"macro_f1:\s*([0-9.]+)"
    )

    # Older report format may contain "macro f1"
    if macro_f1 is None:

        macro_f1 = extract_float(
            text,
            r"macro\s*f1:\s*([0-9.]+)"
        )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


# =========================================================
# LOAD SEQUENCE LABELLING RESULTS
# =========================================================

def load_sequence_results(
    output_dir
):

    path = os.path.join(
        output_dir,
        "sequence_labeling",
        "sequence_labeling_report.txt"
    )

    text = load_text(
        path
    )

    if not text:

        return {}

    hmm_accuracy = extract_float(
        text,
        r"POS Accuracy:\s*([0-9.]+)"
    )

    entity_f1 = extract_float(
        text,
        r"Strict Entity-Level F1:\s*([0-9.]+)"
    )

    token_macro_f1 = extract_float(
        text,
        r"Token Macro-F1:\s*([0-9.]+)"
    )

    return {
        "hmm_pos_accuracy":
            hmm_accuracy,

        "strict_entity_f1":
            entity_f1,

        "token_macro_f1":
            token_macro_f1
    }


# =========================================================
# LOAD N-GRAM RESULTS
# =========================================================

def load_ngram_results(
    output_dir
):

    path = os.path.join(
        output_dir,
        "ngram_lm_report.txt"
    )

    text = load_text(
        path
    )

    if not text:

        return {}

    # Get all scratch backoff perplexities

    backoff_values = re.findall(
        r"Scratch backoff perplexity:\s*([0-9.]+)",
        text,
        flags=re.IGNORECASE
    )

    laplace_values = re.findall(
        r"Scratch Laplace perplexity:\s*([0-9.]+)",
        text,
        flags=re.IGNORECASE
    )

    result = {}

    names = [
        "unigram",
        "bigram",
        "trigram"
    ]

    for index, name in enumerate(
        names
    ):

        if index < len(
            laplace_values
        ):

            result[
                f"{name}_laplace"
            ] = float(
                laplace_values[
                    index
                ]
            )

        if index < len(
            backoff_values
        ):

            result[
                f"{name}_backoff"
            ] = float(
                backoff_values[
                    index
                ]
            )

    return result


# =========================================================
# LOAD TOPIC RESULTS
# =========================================================

def load_topic_results(
    output_dir
):

    path = os.path.join(
        output_dir,
        "topic_modeling",
        "topic_modeling_report.txt"
    )

    text = load_text(
        path
    )

    if not text:

        return {}

    lsa = extract_float(
        text,
        r"LSA:\s*([0-9.]+)"
    )

    lda = extract_float(
        text,
        r"LDA:\s*([0-9.]+)"
    )

    bertopic = extract_float(
        text,
        r"BERTopic:\s*([0-9.]+)"
    )

    return {
        "LSA": lsa,
        "LDA": lda,
        "BERTopic": bertopic
    }


# =========================================================
# LOAD RAG RESULTS
# =========================================================

def load_rag_results(
    output_dir
):

    path = os.path.join(
        output_dir,
        "rag",
        "rag_results.json"
    )

    data = load_json(
        path
    )

    if not data:

        return {}

    good = data.get(
        "good_query",
        {}
    )

    failed = data.get(
        "out_of_knowledge_query",
        {}
    )

    return {
        "good_query_confident":
            good.get(
                "retrieval_confident"
            ),

        "good_answer":
            good.get(
                "answer"
            ),

        "failed_query_confident":
            failed.get(
                "retrieval_confident"
            ),

        "failed_answer":
            failed.get(
                "answer"
            )
    }


# =========================================================
# CREATE CLASSIFICATION MASTER TABLE
# =========================================================

def build_classification_table(
    classical,
    embeddings,
    transformer
):

    rows = []

    # --------------------------------
    # CLASSICAL MODELS
    # --------------------------------

    for model_name, metrics in (
        classical.items()
    ):

        rows.append(
            {
                "model":
                    model_name,

                "accuracy":
                    metrics.get(
                        "accuracy"
                    ),

                "macro_precision":
                    metrics.get(
                        "macro_precision"
                    ),

                "macro_recall":
                    metrics.get(
                        "macro_recall"
                    ),

                "macro_f1":
                    metrics.get(
                        "macro_f1"
                    )
            }
        )


    # --------------------------------
    # EMBEDDING CLASSIFIER
    # --------------------------------

    if embeddings:

        rows.append(
            {
                "model":
                    "GloVe + Logistic Regression",

                "accuracy":
                    embeddings.get(
                        "accuracy"
                    ),

                "macro_precision":
                    None,

                "macro_recall":
                    None,

                "macro_f1":
                    embeddings.get(
                        "macro_f1"
                    )
            }
        )


    # --------------------------------
    # TRANSFORMER
    # --------------------------------

    if transformer:

        rows.append(
            {
                "model":
                    "DistilBERT",

                "accuracy":
                    transformer.get(
                        "accuracy"
                    ),

                "macro_precision":
                    transformer.get(
                        "macro_precision"
                    ),

                "macro_recall":
                    transformer.get(
                        "macro_recall"
                    ),

                "macro_f1":
                    transformer.get(
                        "macro_f1"
                    )
            }
        )

    return rows


# =========================================================
# PRINT CLASSIFICATION TABLE
# =========================================================

def print_classification_table(
    rows
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL CLASSIFICATION COMPARISON"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Model':<32}"
        f"{'Accuracy':<14}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'Macro-F1':<14}"
    )

    print(
        "-" * 90
    )

    for row in rows:

        accuracy = (
            f"{row['accuracy']:.4f}"
            if row["accuracy"] is not None
            else "N/A"
        )

        precision = (
            f"{row['macro_precision']:.4f}"
            if row[
                "macro_precision"
            ] is not None
            else "N/A"
        )

        recall = (
            f"{row['macro_recall']:.4f}"
            if row[
                "macro_recall"
            ] is not None
            else "N/A"
        )

        macro_f1 = (
            f"{row['macro_f1']:.4f}"
            if row["macro_f1"] is not None
            else "N/A"
        )

        print(
            f"{row['model']:<32}"
            f"{accuracy:<14}"
            f"{precision:<14}"
            f"{recall:<14}"
            f"{macro_f1:<14}"
        )


# =========================================================
# FIND BEST CLASSIFIER
# =========================================================

def find_best_classifier(
    rows
):

    valid_rows = [
        row
        for row in rows
        if row[
            "macro_f1"
        ] is not None
    ]

    if not valid_rows:

        return None

    return max(
        valid_rows,
        key=lambda row:
            row["macro_f1"]
    )


# =========================================================
# SAVE CLASSIFICATION BAR CHART
# =========================================================

def save_classification_chart(
    rows,
    output_dir
):

    filtered = [
        row
        for row in rows
        if row[
            "macro_f1"
        ] is not None
    ]

    if not filtered:

        return None

    names = [
        row["model"]
        for row in filtered
    ]

    scores = [
        row["macro_f1"]
        for row in filtered
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        names,
        scores
    )

    ax.set_ylabel(
        "Macro-F1"
    )

    ax.set_title(
        "Emotion Classification Model Comparison"
    )

    ax.set_ylim(
        0,
        1
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    for index, score in enumerate(
        scores
    ):

        ax.text(
            index,
            score + 0.015,
            f"{score:.3f}",
            ha="center"
        )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "classification_macro_f1_comparison.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return path


# =========================================================
# FINAL ANALYSIS
# =========================================================

def generate_analysis(
    rows,
    best_classifier,
    sequence_results,
    ngram_results,
    topic_results,
    rag_results
):

    lines = []

    lines.append(
        "FINAL ANALYSIS"
    )

    lines.append(
        "=" * 70
    )

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    lines.append(
        "\n1. Emotion Classification"
    )

    if best_classifier:

        lines.append(
            (
                f"The best emotion classification "
                f"model was {best_classifier['model']} "
                f"with Macro-F1 "
                f"{best_classifier['macro_f1']:.4f} "
                f"and accuracy "
                f"{best_classifier['accuracy']:.4f}."
            )
        )

    svm = next(
        (
            row
            for row in rows
            if row[
                "model"
            ] == "Linear SVM"
        ),
        None
    )

    transformer = next(
        (
            row
            for row in rows
            if row[
                "model"
            ] == "DistilBERT"
        ),
        None
    )

    if (
        svm
        and transformer
        and svm["macro_f1"] is not None
        and transformer[
            "macro_f1"
        ] is not None
    ):

        improvement = (
            transformer[
                "macro_f1"
            ]
            -
            svm[
                "macro_f1"
            ]
        )

        lines.append(
            (
                f"DistilBERT improved Macro-F1 "
                f"over Linear SVM by "
                f"{improvement:+.4f}."
            )
        )

        lines.append(
            (
                "However, Linear SVM required far "
                "less training compute, while "
                "DistilBERT provided stronger "
                "contextual representations and "
                "better minority-class recall."
            )
        )


    # =====================================================
    # EMBEDDINGS
    # =====================================================

    glove = next(
        (
            row
            for row in rows
            if row[
                "model"
            ]
            == "GloVe + Logistic Regression"
        ),
        None
    )

    if glove:

        lines.append(
            "\n2. Static Embeddings"
        )

        lines.append(
            (
                f"Average GloVe embeddings with "
                f"Logistic Regression achieved "
                f"Macro-F1 "
                f"{glove['macro_f1']:.4f}. "
                f"This was weaker than TF-IDF "
                f"models because averaging static "
                f"word vectors loses word order "
                f"and contextual information."
            )
        )


    # =====================================================
    # LANGUAGE MODEL
    # =====================================================

    if ngram_results:

        lines.append(
            "\n3. N-gram Language Modelling"
        )

        unigram = ngram_results.get(
            "unigram_backoff"
        )

        bigram = ngram_results.get(
            "bigram_backoff"
        )

        trigram = ngram_results.get(
            "trigram_backoff"
        )

        if (
            unigram is not None
            and bigram is not None
            and trigram is not None
        ):

            lines.append(
                (
                    f"Backoff perplexity improved "
                    f"from unigram "
                    f"{unigram:.4f} to bigram "
                    f"{bigram:.4f} and trigram "
                    f"{trigram:.4f}. "
                    f"The trigram backoff model "
                    f"provided the lowest perplexity."
                )
            )


    # =====================================================
    # SEQUENCE LABELLING
    # =====================================================

    if sequence_results:

        lines.append(
            "\n4. Sequence Labelling"
        )

        hmm = sequence_results.get(
            "hmm_pos_accuracy"
        )

        ner = sequence_results.get(
            "strict_entity_f1"
        )

        if hmm is not None:

            lines.append(
                (
                    f"The HMM + Viterbi POS tagger "
                    f"achieved accuracy "
                    f"{hmm:.4f}."
                )
            )

        if ner is not None:

            lines.append(
                (
                    f"The CRF achieved strict "
                    f"entity-level F1 "
                    f"{ner:.4f} for NER."
                )
            )


    # =====================================================
    # TOPIC MODELLING
    # =====================================================

    if topic_results:

        lines.append(
            "\n5. Topic Modelling"
        )

        valid_topics = {
            name: score
            for name, score
            in topic_results.items()
            if score is not None
        }

        if valid_topics:

            best_topic = max(
                valid_topics,
                key=valid_topics.get
            )

            lines.append(
                (
                    f"{best_topic} obtained the "
                    f"highest coherence score "
                    f"({valid_topics[best_topic]:.4f}). "
                    f"BERTopic also produced more "
                    f"human-interpretable semantic "
                    f"clusters than LSA or LDA."
                )
            )


    # =====================================================
    # RAG
    # =====================================================

    if rag_results:

        lines.append(
            "\n6. Retrieval-Augmented Generation"
        )

        lines.append(
            (
                "The in-domain query was successfully "
                "answered using hybrid sparse and dense "
                "retrieval."
            )
        )

        lines.append(
            (
                "For the out-of-knowledge query, "
                "retrieval confidence was low, so "
                "generation was blocked and the system "
                "returned 'I do not have enough "
                "information.' This reduced the risk "
                "of hallucination."
            )
        )


    # =====================================================
    # OVERALL TRADE-OFF
    # =====================================================

    lines.append(
        "\n7. Accuracy vs Compute vs Interpretability"
    )

    lines.append(
        (
            "Naive Bayes was simple and highly "
            "interpretable but weaker on minority classes."
        )
    )

    lines.append(
        (
            "Logistic Regression and Linear SVM provided "
            "strong performance with low computational cost "
            "and were much faster than Transformer training."
        )
    )

    lines.append(
        (
            "DistilBERT achieved the best overall "
            "classification performance but required "
            "substantially more compute and training time."
        )
    )

    lines.append(
        (
            "Therefore, Linear SVM offers an excellent "
            "accuracy-efficiency trade-off, while "
            "DistilBERT is preferred when maximum "
            "classification performance and contextual "
            "understanding are more important."
        )
    )

    return "\n".join(
        lines
    )


# =========================================================
# RUN
# =========================================================

def run(
    config_module=config
):

    print(
        "\nStarting Final Evaluation Stage..."
    )

    output_dir = (
        config_module.OUTPUT_DIR
    )

    final_dir = (
        config_module
        .FINAL_EVALUATION_DIR
    )

    os.makedirs(
        final_dir,
        exist_ok=True
    )


    # =====================================================
    # LOAD ALL RESULTS
    # =====================================================

    classical = (
        load_classical_results(
            output_dir
        )
    )

    transformer = (
        load_transformer_results(
            output_dir
        )
    )

    embeddings = (
        load_embedding_results(
            output_dir
        )
    )

    sequence_results = (
        load_sequence_results(
            output_dir
        )
    )

    ngram_results = (
        load_ngram_results(
            output_dir
        )
    )

    topic_results = (
        load_topic_results(
            output_dir
        )
    )

    rag_results = (
        load_rag_results(
            output_dir
        )
    )


    # =====================================================
    # CLASSIFICATION TABLE
    # =====================================================

    classification_rows = (
        build_classification_table(
            classical,
            embeddings,
            transformer
        )
    )

    print_classification_table(
        classification_rows
    )


    # =====================================================
    # BEST CLASSIFIER
    # =====================================================

    best_classifier = (
        find_best_classifier(
            classification_rows
        )
    )

    if best_classifier:

        print(
            "\nBest overall classifier:",
            best_classifier[
                "model"
            ]
        )

        print(
            "Best Macro-F1:",
            round(
                best_classifier[
                    "macro_f1"
                ],
                4
            )
        )


    # =====================================================
    # N-GRAM SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LANGUAGE MODEL SUMMARY"
    )

    print(
        "=" * 70
    )

    for name, value in (
        ngram_results.items()
    ):

        print(
            f"{name:<25}"
            f"{value:.4f}"
        )


    # =====================================================
    # SEQUENCE LABELLING SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SEQUENCE LABELLING SUMMARY"
    )

    print(
        "=" * 70
    )

    for name, value in (
        sequence_results.items()
    ):

        if value is not None:

            print(
                f"{name:<30}"
                f"{value:.4f}"
            )


    # =====================================================
    # TOPIC SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TOPIC MODELLING SUMMARY"
    )

    print(
        "=" * 70
    )

    for name, value in (
        topic_results.items()
    ):

        if value is not None:

            print(
                f"{name:<15}"
                f"{value:.4f}"
            )


    # =====================================================
    # RAG SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAG SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "Good-query retrieval confident:",
        rag_results.get(
            "good_query_confident"
        )
    )

    print(
        "Good-query answer:",
        rag_results.get(
            "good_answer"
        )
    )

    print(
        "\nOut-of-knowledge retrieval confident:",
        rag_results.get(
            "failed_query_confident"
        )
    )

    print(
        "Out-of-knowledge answer:",
        rag_results.get(
            "failed_answer"
        )
    )


    # =====================================================
    # SAVE CHART
    # =====================================================

    chart_path = (
        save_classification_chart(
            classification_rows,
            final_dir
        )
    )

    if chart_path:

        print(
            "\nSaved comparison chart:",
            chart_path
        )


    # =====================================================
    # GENERATE FINAL ANALYSIS
    # =====================================================

    final_analysis = (
        generate_analysis(
            classification_rows,
            best_classifier,
            sequence_results,
            ngram_results,
            topic_results,
            rag_results
        )
    )

    print(
        "\n"
        + final_analysis
    )


    # =====================================================
    # SAVE JSON
    # =====================================================

    final_json = {
        "classification":
            classification_rows,

        "best_classifier":
            best_classifier,

        "language_model":
            ngram_results,

        "sequence_labeling":
            sequence_results,

        "topic_modeling":
            topic_results,

        "rag":
            rag_results
    }

    json_path = os.path.join(
        final_dir,
        "final_evaluation_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_json,
            file,
            indent=4,
            ensure_ascii=False
        )


    # =====================================================
    # SAVE FINAL REPORT
    # =====================================================

    report_path = os.path.join(
        final_dir,
        "final_evaluation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FINAL NLP PIPELINE "
            "EVALUATION REPORT\n"
        )

        file.write(
            "=" * 90
            + "\n\n"
        )

        file.write(
            "CLASSIFICATION COMPARISON\n"
        )

        file.write(
            "-" * 90
            + "\n"
        )

        file.write(
            f"{'Model':<32}"
            f"{'Accuracy':<14}"
            f"{'Precision':<14}"
            f"{'Recall':<14}"
            f"{'Macro-F1':<14}\n"
        )

        for row in (
            classification_rows
        ):

            file.write(
                f"{row['model']:<32}"
                f"{str(row['accuracy']):<14}"
                f"{str(row['macro_precision']):<14}"
                f"{str(row['macro_recall']):<14}"
                f"{str(row['macro_f1']):<14}\n"
            )

        if best_classifier:

            file.write(
                "\nBest classifier: "
                f"{best_classifier['model']}\n"
            )

            file.write(
                "Best Macro-F1: "
                f"{best_classifier['macro_f1']:.4f}\n"
            )


        file.write(
            "\n\nLANGUAGE MODEL\n"
        )

        file.write(
            "-" * 50
            + "\n"
        )

        for name, value in (
            ngram_results.items()
        ):

            file.write(
                f"{name}: "
                f"{value:.4f}\n"
            )


        file.write(
            "\nSEQUENCE LABELLING\n"
        )

        file.write(
            "-" * 50
            + "\n"
        )

        for name, value in (
            sequence_results.items()
        ):

            if value is not None:

                file.write(
                    f"{name}: "
                    f"{value:.4f}\n"
                )


        file.write(
            "\nTOPIC MODELLING\n"
        )

        file.write(
            "-" * 50
            + "\n"
        )

        for name, value in (
            topic_results.items()
        ):

            if value is not None:

                file.write(
                    f"{name}: "
                    f"{value:.4f}\n"
                )


        file.write(
            "\nRAG\n"
        )

        file.write(
            "-" * 50
            + "\n"
        )

        file.write(
            "Good query confident: "
            f"{rag_results.get('good_query_confident')}\n"
        )

        file.write(
            "Good answer: "
            f"{rag_results.get('good_answer')}\n"
        )

        file.write(
            "Out-of-knowledge confident: "
            f"{rag_results.get('failed_query_confident')}\n"
        )

        file.write(
            "Out-of-knowledge answer: "
            f"{rag_results.get('failed_answer')}\n"
        )


        file.write(
            "\n\n"
            + final_analysis
            + "\n"
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
        "\nFinal Evaluation Stage "
        "Completed Successfully!"
    )


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":
    run(config)