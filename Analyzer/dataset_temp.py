# General
import numpy as np
import pandas as pd

# sklearn
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import _pickle as cPickle
from hazm import *
from collections import Counter
from sklearn.model_selection import train_test_split


# Preprocessing


def get_table(label, reset_index=False):
    path = 'Dataset/' + str(label) + '.csv'
    data = pd.read_csv(path, index_col=None, header=None, encoding="utf-8")
    if reset_index:
        data = data.sample(frac=1).reset_index(drop=True)
    x_data = data[0]
    y_data = data[1]
    cnt = Counter(y_data)
    cnt = dict(cnt)
    print(str(label) + ': ' + str(cnt))
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    return x_data, y_data


def split_dataset(x, y, size=0.25):
    xr, xt, yr, yt = train_test_split(
        x, y, test_size=size, shuffle=True)
    cnt = Counter(y_train)
    cnt = dict(cnt)
    print('new Train: ' + str(cnt))

    cnt = Counter(y_test)
    cnt = dict(cnt)
    print('new Test: ' + str(cnt))
    return xr, xt, yr, yt


# Import & Analyze Dataset

x_test_, y_test_ = get_table('test')
x_original, y_original = get_table('original', True)
x_balanced, y_balanced = get_table('balanced')
x_translation, y_translation = get_table('translation')
x, y = get_table('new_org_test')
x_train, x_test, y_train, y_test = split_dataset(x, y)

# Preprocess


puncs = ['،', '.', ',', ':', ';', '"']
normalizer = Normalizer()
lemmatizer = Lemmatizer()

# Make stop word set

file = pd.read_csv('dataset/per_sw.csv', sep="\n", encoding="utf-8")
stop_set = set(file.values.flatten())


# turn a doc into clean tokens
def clean_doc(doc):
    doc = normalizer.normalize(doc)  # Normalize document using Hazm Normalizer
    tokenized = word_tokenize(doc)  # Tokenize text
    tokens = []
    for t in tokenized:
        temp = t
        for p in puncs:
            temp = temp.replace(p, '')
        tokens.append(temp)
    tokens = [w for w in tokens if not w in stop_set]  # Remove stop words
    tokens = [w for w in tokens if not len(w) <= 1]
    tokens = [w for w in tokens if not w.isdigit()]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  # Lemmatize sentence words using Hazm Lemmatizer
    tokens = ' '.join(tokens)
    return tokens


# Apply preprocessing step to training data
train_docs = np.empty_like(x_train)
for index, document in enumerate(x_train):
    train_docs[index] = clean_doc(document)

# Applying preprocessing step to test data
test_docs = np.empty_like(x_test)
for index, document in enumerate(x_test):
    test_docs[index] = clean_doc(document)
