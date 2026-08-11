import nltk
import spacy

import config

from nltk import pos_tag, word_tokenize
from nltk.tag.mapping import map_tag


# -----------------------------------
# NLTK DOWNLOADS
# -----------------------------------

for resource in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
    "universal_tagset"
]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


# -----------------------------------
# SPACY MODEL
# -----------------------------------

nlp = spacy.load("en_core_web_sm")


# -----------------------------------
# POS TAGGING
# -----------------------------------

def pos_tagging(sentence):

    tokens = word_tokenize(sentence)

    penn_tags = pos_tag(tokens)

    universal_tags = [
        (
            word,
            map_tag(
                "en-ptb",
                "universal",
                tag
            )
        )
        for word, tag in penn_tags
    ]

    return penn_tags, universal_tags


# -----------------------------------
# DEPENDENCY PARSING
# -----------------------------------

def dependency_parse(sentence):

    doc = nlp(sentence)

    triples = []

    for token in doc:

        head = token.head.text
        relation = token.dep_
        dependent = token.text

        triples.append(
            (
                head,
                relation,
                dependent
            )
        )

    return triples


# -----------------------------------
# CONSTITUENCY PARSING
# -----------------------------------

def constituency_parse():

    grammar = nltk.CFG.fromstring("""
        S -> NP VP
        NP -> NNP | DT JJ NN | NP PP
        VP -> VBD NP | VP PP
        PP -> IN NP

        NNP -> 'Google' | 'Berlin'
        VBD -> 'opened'
        DT -> 'a'
        JJ -> 'new'
        NN -> 'office'
        IN -> 'in'
    """)

    parser = nltk.ChartParser(grammar)

    tokens = [
        "Google",
        "opened",
        "a",
        "new",
        "office",
        "in",
        "Berlin"
    ]

    trees = list(parser.parse(tokens))

    if trees:
        return trees[0]

    return "No constituency tree found."


# -----------------------------------
# NER WITH BIO TAGS
# -----------------------------------

def ner_bio_tags(sentence):

    doc = nlp(sentence)

    results = []

    for token in doc:

        if token.ent_iob_ == "O":
            bio_tag = "O"

        else:
            bio_tag = (
                token.ent_iob_
                + "-"
                + token.ent_type_
            )

        results.append(
            (
                token.text,
                bio_tag
            )
        )

    return results


# -----------------------------------
# SIMPLE COREFERENCE DEMO
# -----------------------------------

def coreference_demo():

    sentence = (
        "Sarah bought a new laptop. "
        "She uses it every day."
    )

    explanation = {
        "She": "Sarah",
        "it": "new laptop"
    }

    return sentence, explanation


# -----------------------------------
# RUN
# -----------------------------------

def run(config_module=config):

    print("\nStarting Linguistics Stage...")

    sentence = (
        "Google opened a new office "
        "in Berlin last year."
    )

    print("\n===== EXAMPLE SENTENCE =====")
    print(sentence)

    # POS TAGGING
    penn_tags, universal_tags = pos_tagging(
        sentence
    )

    print("\n===== PENN POS TAGS =====")

    for word, tag in penn_tags:
        print(
            f"{word:<15} {tag}"
        )

    print("\n===== UNIVERSAL POS TAGS =====")

    for word, tag in universal_tags:
        print(
            f"{word:<15} {tag}"
        )

    # DEPENDENCY PARSING
    triples = dependency_parse(
        sentence
    )

    print("\n===== DEPENDENCY PARSE =====")
    print("(HEAD, RELATION, DEPENDENT)")

    for triple in triples:
        print(triple)

    # CONSTITUENCY PARSING
    tree = constituency_parse()

    print("\n===== CONSTITUENCY TREE =====")
    print(tree)

    # NER
    bio_tags = ner_bio_tags(
        sentence
    )

    print("\n===== NER WITH BIO TAGS =====")

    for word, tag in bio_tags:
        print(
            f"{word:<15} {tag}"
        )

    # COREFERENCE
    coref_sentence, coref_map = (
        coreference_demo()
    )

    print("\n===== COREFERENCE DEMO =====")
    print(coref_sentence)

    for pronoun, reference in coref_map.items():

        print(
            f"{pronoun} → {reference}"
        )

    print(
        "\nLinguistics Stage "
        "Completed Successfully!"
    )