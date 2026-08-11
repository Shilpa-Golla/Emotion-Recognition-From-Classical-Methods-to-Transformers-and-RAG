import os

import nltk
import spacy

from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

from sentence_transformers import SentenceTransformer, util

import config


# -----------------------------------
# NLTK DATA
# -----------------------------------

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)


# -----------------------------------
# SPACY MODEL
# -----------------------------------

nlp = spacy.load("en_core_web_sm")


# -----------------------------------
# WORDNET RELATIONS
# -----------------------------------

def wordnet_relations(word):

    synsets = wn.synsets(word)

    if not synsets:
        return {
            "synset": "None",
            "definition": "None",
            "hypernyms": [],
            "hyponyms": [],
            "meronyms": [],
            "antonyms": []
        }

    synset = synsets[0]

    hypernyms = [
        item.name()
        for item in synset.hypernyms()
    ]

    hyponyms = [
        item.name()
        for item in synset.hyponyms()[:5]
    ]

    meronyms = []

    for item in synset.part_meronyms():
        meronyms.append(item.name())

    for item in synset.substance_meronyms():
        meronyms.append(item.name())

    antonyms = []

    for lemma in synset.lemmas():

        for antonym in lemma.antonyms():
            antonyms.append(
                antonym.name()
            )

    return {
        "synset": synset.name(),
        "definition": synset.definition(),
        "hypernyms": hypernyms,
        "hyponyms": hyponyms,
        "meronyms": meronyms,
        "antonyms": antonyms
    }


# -----------------------------------
# LESK WSD
# -----------------------------------

def lesk_wsd(sentence, ambiguous_word):

    tokens = nltk.word_tokenize(sentence)

    sense = lesk(
        tokens,
        ambiguous_word
    )

    if sense is None:
        return None

    return {
        "sense": sense.name(),
        "definition": sense.definition()
    }


# -----------------------------------
# BERT-EMBEDDING WSD
# -----------------------------------

def bert_wsd(
    sentence,
    ambiguous_word,
    model
):

    possible_senses = wn.synsets(
        ambiguous_word
    )

    if not possible_senses:
        return None

    # Encode the complete sentence
    sentence_embedding = model.encode(
        sentence,
        convert_to_tensor=True
    )

    best_score = -1
    best_synset = None

    for synset in possible_senses:

        # WordNet definition / gloss
        gloss = synset.definition()

        gloss_embedding = model.encode(
            gloss,
            convert_to_tensor=True
        )

        similarity = util.cos_sim(
            sentence_embedding,
            gloss_embedding
        ).item()

        if similarity > best_score:
            best_score = similarity
            best_synset = synset

    return {
        "sense": best_synset.name(),
        "definition": best_synset.definition(),
        "similarity": best_score
    }


# -----------------------------------
# SEMANTIC ROLE DEMO
# -----------------------------------

def semantic_roles(sentence):

    doc = nlp(sentence)

    roles = {
        "Agent": None,
        "Patient/Theme": None,
        "Recipient": None
    }

    for token in doc:

        # Subject performing the action
        if token.dep_ in ["nsubj", "nsubjpass"]:

            roles["Agent"] = " ".join(
                [
                    child.text
                    for child in token.subtree
                ]
            )

        # Direct object
        elif token.dep_ in ["dobj", "obj"]:

            roles["Patient/Theme"] = " ".join(
                [
                    child.text
                    for child in token.subtree
                ]
            )

        # Indirect object / dative
        elif token.dep_ in ["dative", "iobj"]:

            roles["Recipient"] = " ".join(
                [
                    child.text
                    for child in token.subtree
                ]
            )

    return roles


# -----------------------------------
# RUN
# -----------------------------------

def run(config_module=config):

    print("\nStarting Semantics Stage...")

    report_lines = []

    # ===================================
    # PART 1: WORDNET
    # ===================================

    word = config_module.SEMANTICS_WORD

    relations = wordnet_relations(
        word
    )

    print("\n===== WORDNET RELATIONS =====")
    print("Word:", word)

    print(
        "Synset:",
        relations["synset"]
    )

    print(
        "Definition:",
        relations["definition"]
    )

    print(
        "Hypernyms:",
        relations["hypernyms"]
    )

    print(
        "Hyponyms:",
        relations["hyponyms"]
    )

    print(
        "Meronyms:",
        relations["meronyms"]
    )

    print(
        "Antonyms:",
        relations["antonyms"]
        if relations["antonyms"]
        else "None found"
    )

    report_lines.append(
        "===== WORDNET RELATIONS ====="
    )

    report_lines.append(
        f"Word: {word}"
    )

    report_lines.append(
        f"Synset: {relations['synset']}"
    )

    report_lines.append(
        f"Definition: "
        f"{relations['definition']}"
    )

    report_lines.append(
        f"Hypernyms: "
        f"{relations['hypernyms']}"
    )

    report_lines.append(
        f"Hyponyms: "
        f"{relations['hyponyms']}"
    )

    report_lines.append(
        f"Meronyms: "
        f"{relations['meronyms']}"
    )

    report_lines.append(
        f"Antonyms: "
        f"{relations['antonyms']}"
    )

    # ===================================
    # PART 2: WORD SENSE DISAMBIGUATION
    # ===================================

    print(
        "\n===== WORD SENSE "
        "DISAMBIGUATION ====="
    )

    sentence_1 = (
        "I deposited my money "
        "in the bank."
    )

    sentence_2 = (
        "We sat on the bank "
        "of the river."
    )

    ambiguous_word = "bank"

    model = SentenceTransformer(
        config_module
        .SENTENCE_TRANSFORMER_MODEL
    )

    for sentence in [
        sentence_1,
        sentence_2
    ]:

        print("\nSentence:")
        print(sentence)

        lesk_result = lesk_wsd(
            sentence,
            ambiguous_word
        )

        bert_result = bert_wsd(
            sentence,
            ambiguous_word,
            model
        )

        print("\nLesk result:")

        if lesk_result:

            print(
                "Sense:",
                lesk_result["sense"]
            )

            print(
                "Definition:",
                lesk_result["definition"]
            )

        else:
            print("No sense found.")

        print(
            "\nBERT-Embedding result:"
        )

        if bert_result:

            print(
                "Sense:",
                bert_result["sense"]
            )

            print(
                "Definition:",
                bert_result["definition"]
            )

            print(
                "Similarity:",
                round(
                    bert_result[
                        "similarity"
                    ],
                    4
                )
            )

        report_lines.append("")
        report_lines.append(
            "===== WSD ====="
        )

        report_lines.append(
            f"Sentence: {sentence}"
        )

        report_lines.append(
            f"Lesk: {lesk_result}"
        )

        report_lines.append(
            f"BERT: {bert_result}"
        )

    # ===================================
    # PART 3: SEMANTIC ROLES
    # ===================================

    role_sentence = (
        "Sarah gave John a book."
    )

    roles = semantic_roles(
        role_sentence
    )

    print(
        "\n===== SEMANTIC ROLE "
        "LABELLING ====="
    )

    print(role_sentence)

    print(
        "Agent:",
        roles["Agent"]
    )

    print(
        "Patient/Theme:",
        roles["Patient/Theme"]
    )

    print(
        "Recipient:",
        roles["Recipient"]
    )

    report_lines.append("")
    report_lines.append(
        "===== SEMANTIC ROLES ====="
    )

    report_lines.append(
        f"Sentence: {role_sentence}"
    )

    report_lines.append(
        f"Agent: {roles['Agent']}"
    )

    report_lines.append(
        "Patient/Theme: "
        f"{roles['Patient/Theme']}"
    )

    report_lines.append(
        f"Recipient: "
        f"{roles['Recipient']}"
    )

    # ===================================
    # SAVE RESULTS
    # ===================================

    os.makedirs(
        config_module.OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        config_module.OUTPUT_DIR,
        "semantics_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report_lines)
        )

    print(
        "\nSaved:",
        output_path
    )

    print(
        "\nSemantics Stage "
        "Completed Successfully!"
    )