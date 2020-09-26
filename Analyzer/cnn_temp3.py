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
from keras.models import model_from_json

import codecs

# Keras
from keras import optimizers
from keras.models import Model, Sequential, save_model, load_model
from keras.layers import Dense, Input, Embedding, Dropout
from keras.layers import GlobalMaxPool1D, MaxPooling1D, GlobalMaxPooling1D

from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from keras import utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
# Preprocessing

from hazm import *
# Visualization

import matplotlib.pyplot as plt
from keras.utils import plot_model

with open('dataset/x_test.pkl', 'rb') as handle:
    x_test = cPickle.load(handle)

with open('dataset/y_test.pkl', 'rb') as handle:
    y_test = cPickle.load(handle)

cnt = Counter(y_test)
cnt = dict(cnt)
class_names = {k: v for k, v in sorted(cnt.items(), key=lambda item: item[1], reverse=True)}
print('test: ' + str(class_names))

with open('dataset/x_train.pkl', 'rb') as handle:
    x_train = cPickle.load(handle)

with open('dataset/y_train.pkl', 'rb') as handle:
    y_train = cPickle.load(handle)

cnt = Counter(y_train)
cnt = dict(cnt)
class_names = {k: v for k, v in sorted(cnt.items(), key=lambda item: item[1], reverse=True)}
print('train: ' + str(class_names))

# Preprocess


puncs = ['،', '.', ',', ':', ';', '"']
normalizer = Normalizer()
lemmatizer = Lemmatizer()

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

with open('dataset/test_docs.pkl', 'rb') as handle:
    test_docs = cPickle.load(handle)

with open('dataset/train_docs.pkl', 'rb') as handle:
    train_docs = cPickle.load(handle)

# num_words = 2000


with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = cPickle.load(handle)

# Find maximum length of training sentences
max_length = max([len(s.split()) for s in train_docs])

# Embed training sequences
encoded_docs = tokenizer.texts_to_sequences(train_docs)

# Pad embeded training sequences
x_train_padded = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# Define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index)

# Embed testing sequences
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# Pad testing sequences
x_test_padded = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# Prepare labels for categorical prediction
categorical_y_train = to_categorical(y_train + 2, 5)
categorical_y_test = to_categorical(y_test + 2, 5)

model_cnn = load_model('model/CNN_classifier25.h5')

loss_cnn, acc_cnn = model_cnn.evaluate(x_test_padded, categorical_y_test, verbose=0)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc_cnn))

# class_names = {k: v for k, v in sorted(cnt.items(), key=lambda item: item[1], reverse=True)}
# print(class_names)

# loss, acc = model_cnn.evaluate(x_test_padded[237:247], categorical_y_test[237:247])
# print("Sample Trained model, accuracy: {:5.2f}%".format(100 * acc))

# for i in range(237, 247):
#     predictions = model_cnn.predict(tf.expand_dims(x_test_padded[i], 0))
#     classes = np.argmax(predictions, axis=1)
#     print("real: " + str(y_test[i]) + " ,  label : " + str(classes[0] - 2) +
#           ' ,  cat : ' + str(categorical_y_test[i])
#           + " ,  predict: " + str((predictions[0] * 100).astype(int)))
# print(x_test[237])
# print(x_test_padded[310])
