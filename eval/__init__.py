from nltk.metrics.scores import accuracy, precision, recall, f_measure


def process_test_data(filename):
    name_entities = []
    with open(filename) as f:
        for line in f:
            try:
                token, _, _, tag = line.strip().split()
            except:
                pass
            else:
                if "LOC" in tag:
                    tag = "LOCATION"
                else:
                    tag = "O"
                name_entities.append((token, tag))
    return name_entities


def extract_sents(filename):
    sents = []
    current_sent = []
    with open(filename) as f:
        for line in f:
            try:
                token, _, _, tag = line.strip().split()
            except:
                pass
            else:
                current_sent.append(token)
                if token == "." and tag == "O":
                    sents.append(current_sent)
                    current_sent = []
    return sents


def annotate(tokenized_sents, annotate_method):
    annotations = []
    count = len(tokenized_sents)
    for (index, tokenize_sent) in enumerate(tokenized_sents):
        print(f"processing {index}/{count} sentenses")
        result = annotate_method(tokenize_sent)
        annotations.extend(result)
    return annotations


def get_measures(reference, test):
    acc = accuracy(reference, test)
    reference_set = set(reference)
    test_set = set(test)
    pre = precision(reference_set, test_set)
    rec = recall(reference_set, test_set)
    f = f_measure(reference_set, test_set)
    return acc, pre, rec, f


def save_result(reference, test):
    with open("eval.txt", "w") as f:
        f.write("reference\n")
        list(map(lambda item: f.write(" ".join(item) + "\n"), reference))
        f.write("test\n")
        list(map(lambda item: f.write(" ".join(item) + "\n"), test))


def evaluate(dataset, recognizer):
    name_entities = process_test_data(dataset)
    sents = extract_sents(dataset)
    if recognizer == "nltk":
        from .eval_nltk import annotate_org
    else:
        from .eval_stanford import annotate_org
    predicted_name_entities = annotate(sents, annotate_org)
    save_result(name_entities, predicted_name_entities)
    acc, pre, rec, f = get_measures(name_entities, predicted_name_entities)
    print(f"nltk evaluation on {dataset}, accuracy={acc}, precision={pre}, recall={rec}, f_measure={f}")
