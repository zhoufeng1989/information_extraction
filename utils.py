from nltk import sent_tokenize, word_tokenize


def tokenize(filename):
    text = open(filename).read()
    sents = sent_tokenize(text)
    tokenized_sents = [word_tokenize(sent) for sent in sents]
    return tokenized_sents


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
