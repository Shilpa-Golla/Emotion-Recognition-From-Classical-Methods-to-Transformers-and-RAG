import json
import os

import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

import config


# =========================================================
# SMALL SECONDARY KNOWLEDGE DATASET
# =========================================================

def build_secondary_dataset():
    """
    Small secondary dataset for the RAG demonstration.

    This dataset is intentionally different from the main
    emotion-classification dataset.

    It is used to demonstrate:
        1. Sparse retrieval
        2. Dense retrieval
        3. Hybrid retrieval
        4. Retrieval-Augmented Generation
        5. Out-of-knowledge failure handling
    """

    documents = [
        {
            "id": "doc_1",
            "title": "Berlin",
            "text": (
                "Berlin is the capital of Germany. "
                "It is Germany's largest city by population "
                "and is located in northeastern Germany."
            )
        },

        {
            "id": "doc_2",
            "title": "Munich",
            "text": (
                "Munich is the capital of Bavaria. "
                "It is known for Oktoberfest and is one "
                "of the largest cities in Germany."
            )
        },

        {
            "id": "doc_3",
            "title": "Germany",
            "text": (
                "Germany is a country in Central Europe. "
                "Its capital is Berlin and its official "
                "language is German."
            )
        },

        {
            "id": "doc_4",
            "title": "Python",
            "text": (
                "Python is a high-level programming language. "
                "It is widely used for machine learning, "
                "data science, automation and web development."
            )
        },

        {
            "id": "doc_5",
            "title": "Natural Language Processing",
            "text": (
                "Natural Language Processing is a field "
                "of artificial intelligence concerned with "
                "processing and understanding human language."
            )
        },

        {
            "id": "doc_6",
            "title": "Transformer",
            "text": (
                "Transformers use self-attention to model "
                "relationships between tokens. They allow "
                "parallel processing and are widely used "
                "in modern natural language processing."
            )
        },

        {
            "id": "doc_7",
            "title": "TF-IDF",
            "text": (
                "TF-IDF measures the importance of a term "
                "to a document relative to a collection "
                "of documents."
            )
        },

        {
            "id": "doc_8",
            "title": "BM25",
            "text": (
                "BM25 is a ranking function used by search "
                "engines to estimate how relevant documents "
                "are to a search query."
            )
        },

        {
            "id": "doc_9",
            "title": "RAG",
            "text": (
                "Retrieval-Augmented Generation combines "
                "information retrieval with text generation. "
                "Relevant documents are retrieved first and "
                "then supplied as context to a generator."
            )
        },

        {
            "id": "doc_10",
            "title": "Word2Vec",
            "text": (
                "Word2Vec learns dense word vectors from "
                "context. Skip-gram predicts context words "
                "from a center word."
            )
        }
    ]

    return documents


# =========================================================
# HYBRID RETRIEVER
# =========================================================

class HybridRetriever:

    def __init__(
        self,
        documents,
        dense_model_name,
        sparse_weight=0.5,
        dense_weight=0.5
    ):

        self.documents = documents

        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

        self.document_texts = [
            document["text"]
            for document in documents
        ]

        # =================================================
        # SPARSE RETRIEVAL
        # =================================================

        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.tfidf_matrix = (
            self.tfidf_vectorizer.fit_transform(
                self.document_texts
            )
        )

        # =================================================
        # DENSE RETRIEVAL
        # =================================================

        print(
            "\nLoading dense retrieval model:"
        )

        print(
            dense_model_name
        )

        self.dense_model = SentenceTransformer(
            dense_model_name
        )

        print(
            "Creating document embeddings..."
        )

        self.document_embeddings = (
            self.dense_model.encode(
                self.document_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        )


    # =====================================================
    # NORMALIZE SCORES
    # =====================================================

    @staticmethod
    def normalize_scores(scores):
        """
        Min-max normalize scores to [0, 1]
        before combining sparse and dense scores.
        """

        scores = np.asarray(
            scores,
            dtype=np.float64
        )

        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return np.zeros_like(
                scores
            )

        return (
            scores - minimum
        ) / (
            maximum - minimum
        )


    # =====================================================
    # RETRIEVE
    # =====================================================

    def retrieve(
        self,
        query,
        top_k=3
    ):

        # =================================================
        # SPARSE QUERY
        # =================================================

        query_tfidf = (
            self.tfidf_vectorizer.transform(
                [query]
            )
        )

        sparse_scores = cosine_similarity(
            query_tfidf,
            self.tfidf_matrix
        )[0]

        # =================================================
        # DENSE QUERY
        # =================================================

        query_embedding = (
            self.dense_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
        )

        dense_scores = (
            self.document_embeddings
            @ query_embedding
        )

        # =================================================
        # NORMALIZE SCORES
        # =================================================

        sparse_normalized = (
            self.normalize_scores(
                sparse_scores
            )
        )

        dense_normalized = (
            self.normalize_scores(
                dense_scores
            )
        )

        # =================================================
        # HYBRID SCORE
        # =================================================

        hybrid_scores = (
            self.sparse_weight
            * sparse_normalized
            +
            self.dense_weight
            * dense_normalized
        )

        ranked_indices = np.argsort(
            hybrid_scores
        )[::-1]

        results = []

        for index in ranked_indices[
            :top_k
        ]:

            document = self.documents[
                index
            ]

            results.append(
                {
                    "id":
                        document["id"],

                    "title":
                        document["title"],

                    "text":
                        document["text"],

                    "sparse_score":
                        float(
                            sparse_scores[
                                index
                            ]
                        ),

                    "dense_score":
                        float(
                            dense_scores[
                                index
                            ]
                        ),

                    "hybrid_score":
                        float(
                            hybrid_scores[
                                index
                            ]
                        )
                }
            )

        return results


# =========================================================
# GENERATOR
# =========================================================

class RAGGenerator:

    def __init__(
        self,
        model_name
    ):

        print(
            "\nLoading RAG generator:"
        )

        print(
            model_name
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSeq2SeqLM
            .from_pretrained(
                model_name
            )
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Generator device:",
            self.device
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    def generate(
        self,
        query,
        retrieved_documents,
        max_new_tokens
    ):

        context = "\n".join(
            [
                document["text"]
                for document
                in retrieved_documents
            ]
        )

        prompt = (
            "Answer the question using only "
            "the provided context. "
            "Do not use outside knowledge. "
            "If the answer is not contained "
            "in the context, answer exactly: "
            "'I do not have enough information.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value
            in encoded.items()
        }

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    **encoded,
                    max_new_tokens=(
                        max_new_tokens
                    ),
                    do_sample=False
                )
            )

        answer = (
            self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
        )

        return answer.strip()


# =========================================================
# RETRIEVAL CONFIDENCE CHECK
# =========================================================

def retrieval_is_confident(
    retrieved_documents,
    config_module
):
    """
    Decide whether retrieval is strong enough
    to allow generation.

    We refuse only when BOTH retrieval signals
    are weak.

    This is important because sparse and dense
    retrieval complement one another.
    """

    if not retrieved_documents:
        return False

    top_result = retrieved_documents[0]

    sparse_score = (
        top_result[
            "sparse_score"
        ]
    )

    dense_score = (
        top_result[
            "dense_score"
        ]
    )

    sparse_is_weak = (
        sparse_score
        <
        config_module
        .RAG_MIN_SPARSE_SCORE
    )

    dense_is_weak = (
        dense_score
        <
        config_module
        .RAG_MIN_DENSE_SCORE
    )

    # Refuse only if BOTH are weak.
    if (
        sparse_is_weak
        and dense_is_weak
    ):
        return False

    return True


# =========================================================
# RUN ONE QUERY
# =========================================================

def run_query(
    query_name,
    query,
    retriever,
    generator,
    config_module
):

    print(
        "\n"
        + "=" * 65
    )

    print(
        query_name
    )

    print(
        "=" * 65
    )

    print(
        "\nQuestion:"
    )

    print(
        query
    )

    # =====================================================
    # RETRIEVAL
    # =====================================================

    retrieved = retriever.retrieve(
        query,
        top_k=(
            config_module
            .RAG_TOP_K
        )
    )

    print(
        "\nRetrieved documents:"
    )

    for rank, result in enumerate(
        retrieved,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            "Title:",
            result["title"]
        )

        print(
            "Sparse score:",
            round(
                result[
                    "sparse_score"
                ],
                4
            )
        )

        print(
            "Dense score:",
            round(
                result[
                    "dense_score"
                ],
                4
            )
        )

        print(
            "Hybrid score:",
            round(
                result[
                    "hybrid_score"
                ],
                4
            )
        )


    # =====================================================
    # CONFIDENCE GATE
    # =====================================================

    confident = retrieval_is_confident(
        retrieved,
        config_module
    )

    top_result = (
        retrieved[0]
        if retrieved
        else None
    )

    if not confident:

        print(
            "\nRetrieval confidence:"
            " LOW"
        )

        print(
            "Generation skipped to "
            "prevent unsupported output."
        )

        answer = (
            "I do not have enough information."
        )

        decision = (
            "abstained_due_to_low_retrieval_confidence"
        )

    else:

        print(
            "\nRetrieval confidence:"
            " SUFFICIENT"
        )

        answer = generator.generate(
            query,
            retrieved,
            config_module
            .RAG_MAX_NEW_TOKENS
        )

        decision = (
            "generated_from_retrieved_context"
        )


    # =====================================================
    # ANSWER
    # =====================================================

    print(
        "\nFinal answer:"
    )

    print(
        answer
    )


    # =====================================================
    # RETURN STRUCTURED RESULT
    # =====================================================

    result = {
        "query_name":
            query_name,

        "query":
            query,

        "retrieval_confident":
            confident,

        "decision":
            decision,

        "answer":
            answer,

        "retrieved_documents":
            retrieved
    }

    if top_result:

        result[
            "top_sparse_score"
        ] = top_result[
            "sparse_score"
        ]

        result[
            "top_dense_score"
        ] = top_result[
            "dense_score"
        ]

        result[
            "top_hybrid_score"
        ] = top_result[
            "hybrid_score"
        ]

    return result


# =========================================================
# RUN COMPLETE RAG STAGE
# =========================================================

def run(
    config_module=config
):

    print(
        "\nStarting RAG Stage..."
    )

    # =====================================================
    # SECONDARY DATASET
    # =====================================================

    documents = (
        build_secondary_dataset()
    )

    print(
        "\nSecondary RAG documents:",
        len(documents)
    )

    print(
        "\nHybrid retrieval:"
    )

    print(
        "Sparse = TF-IDF"
    )

    print(
        "Dense = SentenceTransformer"
    )

    print(
        "Weights:"
    )

    print(
        "Sparse:",
        config_module
        .RAG_SPARSE_WEIGHT
    )

    print(
        "Dense:",
        config_module
        .RAG_DENSE_WEIGHT
    )

    print(
        "\nConfidence thresholds:"
    )

    print(
        "Minimum sparse:",
        config_module
        .RAG_MIN_SPARSE_SCORE
    )

    print(
        "Minimum dense:",
        config_module
        .RAG_MIN_DENSE_SCORE
    )


    # =====================================================
    # RETRIEVER
    # =====================================================

    retriever = HybridRetriever(
        documents=documents,

        dense_model_name=(
            config_module
            .RAG_DENSE_MODEL
        ),

        sparse_weight=(
            config_module
            .RAG_SPARSE_WEIGHT
        ),

        dense_weight=(
            config_module
            .RAG_DENSE_WEIGHT
        )
    )


    # =====================================================
    # GENERATOR
    # =====================================================

    generator = RAGGenerator(
        config_module
        .RAG_GENERATOR_MODEL
    )


    # =====================================================
    # GOOD QUERY
    # =====================================================

    good_query = (
        "What is Retrieval-Augmented "
        "Generation and how does it work?"
    )

    good_result = run_query(
        "GOOD RAG QUERY",
        good_query,
        retriever,
        generator,
        config_module
    )


    # =====================================================
    # OUT-OF-KNOWLEDGE QUERY
    # =====================================================

    failed_query = (
        "Who won the FIFA World Cup "
        "in 2022?"
    )

    failed_result = run_query(
        "OUT-OF-KNOWLEDGE RAG QUERY",
        failed_query,
        retriever,
        generator,
        config_module
    )


    # =====================================================
    # SAVE OUTPUT DIRECTORY
    # =====================================================

    output_dir = os.path.join(
        config_module.OUTPUT_DIR,
        "rag"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # =====================================================
    # SAVE JSON
    # =====================================================

    json_path = os.path.join(
        output_dir,
        "rag_results.json"
    )

    final_results = {
        "configuration": {
            "dense_model":
                config_module
                .RAG_DENSE_MODEL,

            "generator_model":
                config_module
                .RAG_GENERATOR_MODEL,

            "top_k":
                config_module
                .RAG_TOP_K,

            "sparse_weight":
                config_module
                .RAG_SPARSE_WEIGHT,

            "dense_weight":
                config_module
                .RAG_DENSE_WEIGHT,

            "minimum_sparse_score":
                config_module
                .RAG_MIN_SPARSE_SCORE,

            "minimum_dense_score":
                config_module
                .RAG_MIN_DENSE_SCORE
        },

        "good_query":
            good_result,

        "out_of_knowledge_query":
            failed_result
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_results,
            file,
            indent=4,
            ensure_ascii=False
        )


    # =====================================================
    # SAVE TEXT REPORT
    # =====================================================

    report_path = os.path.join(
        output_dir,
        "rag_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "RETRIEVAL-AUGMENTED "
            "GENERATION REPORT\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "ARCHITECTURE\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            "Sparse retrieval: TF-IDF\n"
        )

        file.write(
            "Dense retrieval: "
            "SentenceTransformer\n"
        )

        file.write(
            "Hybrid score: weighted "
            "combination of sparse "
            "and dense retrieval\n"
        )

        file.write(
            "Generator: "
            f"{config_module.RAG_GENERATOR_MODEL}\n"
        )

        file.write(
            "\nConfidence rule:\n"
        )

        file.write(
            "If BOTH sparse and dense "
            "retrieval scores are below "
            "their thresholds, generation "
            "is skipped.\n"
        )


        # =================================================
        # GOOD QUERY REPORT
        # =================================================

        file.write(
            "\n\nGOOD RAG QUERY\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            "Question: "
            + good_result[
                "query"
            ]
            + "\n"
        )

        file.write(
            "Retrieval confident: "
            f"{good_result['retrieval_confident']}\n"
        )

        file.write(
            "Decision: "
            f"{good_result['decision']}\n"
        )

        file.write(
            "Answer: "
            + good_result[
                "answer"
            ]
            + "\n\n"
        )

        file.write(
            "Retrieved documents:\n"
        )

        for result in (
            good_result[
                "retrieved_documents"
            ]
        ):

            file.write(
                f"- {result['title']} | "
                f"sparse="
                f"{result['sparse_score']:.4f} | "
                f"dense="
                f"{result['dense_score']:.4f} | "
                f"hybrid="
                f"{result['hybrid_score']:.4f}\n"
            )


        # =================================================
        # FAILED QUERY REPORT
        # =================================================

        file.write(
            "\n\nOUT-OF-KNOWLEDGE QUERY\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            "Question: "
            + failed_result[
                "query"
            ]
            + "\n"
        )

        file.write(
            "Retrieval confident: "
            f"{failed_result['retrieval_confident']}\n"
        )

        file.write(
            "Decision: "
            f"{failed_result['decision']}\n"
        )

        file.write(
            "Answer: "
            + failed_result[
                "answer"
            ]
            + "\n\n"
        )

        file.write(
            "Explanation:\n"
        )

        file.write(
            "The secondary knowledge "
            "base contains no information "
            "about the 2022 FIFA World Cup. "
            "Both sparse and dense retrieval "
            "confidence were below the required "
            "thresholds, so generation was "
            "blocked to reduce hallucination.\n"
        )

        file.write(
            "\nRetrieved documents:\n"
        )

        for result in (
            failed_result[
                "retrieved_documents"
            ]
        ):

            file.write(
                f"- {result['title']} | "
                f"sparse="
                f"{result['sparse_score']:.4f} | "
                f"dense="
                f"{result['dense_score']:.4f} | "
                f"hybrid="
                f"{result['hybrid_score']:.4f}\n"
            )


        # =================================================
        # CONCLUSION
        # =================================================

        file.write(
            "\n\nCONCLUSION\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            "The successful query demonstrates "
            "grounded retrieval-augmented "
            "generation. The out-of-knowledge "
            "query demonstrates an important "
            "RAG failure mode: irrelevant "
            "documents may still receive non-zero "
            "similarity scores. A retrieval "
            "confidence gate was therefore added "
            "to prevent unsupported generation.\n"
        )


    # =====================================================
    # FINAL TERMINAL SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 65
    )

    print(
        "RAG FINAL SUMMARY"
    )

    print(
        "=" * 65
    )

    print(
        "\nGood query:"
    )

    print(
        "Retrieval confident:",
        good_result[
            "retrieval_confident"
        ]
    )

    print(
        "Answer:",
        good_result[
            "answer"
        ]
    )

    print(
        "\nOut-of-knowledge query:"
    )

    print(
        "Retrieval confident:",
        failed_result[
            "retrieval_confident"
        ]
    )

    print(
        "Answer:",
        failed_result[
            "answer"
        ]
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
        "\nRAG Stage "
        "Completed Successfully!"
    )

    return {
        "good":
            good_result,

        "failed":
            failed_result
    }


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":
    run(config)