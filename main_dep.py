from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def lemmatize(item):
    new_item = []
    for entry in item:
        if isinstance(entry, tuple) and (entry[1].startswith("VB")):
            new_entry = (lemmatizer.lemmatize(entry[0], "v"), entry[1])
        else:
            new_entry = entry
        new_item.append(new_entry)
    return tuple(new_item)


if __name__ == "__main__":
    import os
    import sys
    import json
    from nltk.stem import WordNetLemmatizer
    from collections import defaultdict, Counter
    from dep.dep_parser import dep_parse
    directory = sys.argv[1]
    filenames = os.listdir(directory)
    for filename in filenames:
        entity_result = defaultdict(list)
        entity = filename.rsplit(".", 1)[0]
        filepath = os.path.join(directory, filename)
        sents = json.loads(open(filepath).read())

        print(f"Parse verbs for entity {entity}")
        items = map(lambda item: lemmatize(item), dep_parse(sents, entity))
        for (governor, dep, dependent) in items:
            if dependent[0] in entity:
                verb = governor[0]
            else:
                verb = dependent[0]
            entity_result[dep].append(verb)
        for (dep, verbs) in entity_result.items():
            print(f"dep,{dep}")
            counter = Counter(verbs)
            for (verb, count) in counter.most_common():
                print(f"{verb}:{count}")
            print("--" * 20)
