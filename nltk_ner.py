from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from utils import tokenize


def nltk_ner(filename):
    tokenized_sents = tokenize(filename)
    orgs = []
    for (index, words) in enumerate(tokenized_sents):
        _orgs = extract_orgs(words)
        orgs.extend([(index, org) for org in _orgs])
    return orgs


def extract_orgs(words):
    orgs = []
    pos_words = pos_tag(words)
    for trunk in ne_chunk(pos_words):
        if isinstance(trunk, Tree):
            label = trunk.label()
            if label == "ORGANIZATION":
                org = " ".join([l[0] for l in trunk])
                orgs.append(org)
    return orgs
