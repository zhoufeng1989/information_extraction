# information_extraction
An information extraction task with NLTK

# NER task 
In this repo, there are four ways to do NER tasks.

* nltk (nltk required)
`python main\_ner.py data/task nltk`


* nltk enhancement with combining continuous named entities. 
`python main\_ner.py data/task enltk`


* spacy (spacy required)
Download spacy model first. 
`python -m spacy download en`
`python main\_ner.py data/task spacy`


* Stanford NER (Java 8 required)

`export STANFORD_MODE="jars/english.all.3class.distsim.crf.ser.gz"`
`export STANFORD_JAR="jars/stanford-corenlp-3.9.1.jar"`
`python main\_ner.py data/task stanford`


# NER evaluation 
There is one dataset to evaluate the performance of ners. 

* Eval nltk  
`python main\_eval.py data/conll2003.txt nltk`

* Eval spacy 
`python main\_eval.py data/conll2003.txt spacy`

* Eval stanford 
`export STANFORD_MODE="jars/english.all.3class.distsim.crf.ser.gz"`
`export STANFORD_JAR="jars/stanford-corenlp-3.9.1.jar"`
`python main\_eval.py data/conll2003.txt stanford`
