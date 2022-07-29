import nltk
# nltk.download('punkt')
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation
import pymorphy2
import numpy as np


russian_stopwords = stopwords.words("russian")
morpher = pymorphy2.MorphAnalyzer()
stemmer = SnowballStemmer(language="russian")


def tokenize(sentence: str, stop_words: bool = False):
    tokens = nltk.word_tokenize(sentence.lower())
    if stop_words:
        return [token for token in tokens if token not in russian_stopwords and token not in punctuation]
    else:
        return [token for token in tokens if token not in punctuation]


def stem(word):
    return stemmer.stem(word.lower())


def morph(word):
    return morpher.parse(word)[0].normal_form


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [morph(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag


