from nltk.tag import StanfordNERTagger
from utils import tokenize
import os


def stanford_ner(filename):
    tokenized_sents = tokenize(filename)
    orgs = []
    model_dir = os.getenv("STANFORD_MODEL")
    jar_dir = os.getenv("STANFORD_JAR")
    stanford_tagger = StanfordNERTagger(model_dir, jar_dir)
    for words in tokenized_sents:
        orgs.extend(extract_orgs(words, stanford_tagger))
    return orgs


def extract_orgs(words, ner_tagger):
        current = []
        orgs = []
        for (word, label) in ner_tagger.tag(words):
            if label == "ORGANIZATION":
                current.append(word)
            elif label == "O" and current:
                org = " ".join(current)
                orgs.append(org)
                current = []
        if current:
            org = " ".join(current)
            orgs.append(org)
        return orgs
