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
# Measuring metrics
from sklearn.metrics import f1_score

# Import & Analyze Dataset
test = pd.read_csv('Dataset/test.csv', index_col=None, header=None, encoding="utf-8")

x_test = test[0]
y_test = test[1]

cnt = Counter(y_test)
cnt = dict(cnt)
print('test: ' + str(cnt))

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

original = pd.read_csv('Dataset/original.csv', index_col=None, header=None, encoding="utf-8")
balanced = pd.read_csv('Dataset/balanced.csv', index_col=None, header=None, encoding="utf-8")
translation = pd.read_csv('dataset/translation.csv', index_col=None, header=None, encoding="utf-8")

selected_dataset = original

selected_dataset = selected_dataset.sample(frac=1).reset_index(drop=True)

x_train = selected_dataset[0]
y_train = selected_dataset[1]

cnt = Counter(y_train)
cnt = dict(cnt)
print('train: ' + str(cnt))

# Convert dataframes to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# Preprocess


puncs = ['،', '.', ',', ':', ';', '"']
normalizer = Normalizer()
lemmatizer = Lemmatizer()


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
    # tokens = [w for w in tokens if not w in stop_set]    # Remove stop words
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

num_words = 2000

# Create the tokenizer
tokenizer = Tokenizer(num_words=num_words)

# fFt the tokenizer on the training documents
tokenizer.fit_on_texts(train_docs)

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
categorical_y_train = to_categorical(y_train, 5)
categorical_y_test = to_categorical(y_test, 5)

new_model = load_model('my_model_weights.h5')

new_model.summary()
new_model.get_weights()
print(new_model.optimizer)

loss_cnn, acc_cnn = new_model.evaluate(x_test_padded, categorical_y_test, verbose=0)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc_cnn))

class_names = {k: v for k, v in sorted(cnt.items(), key=lambda item: item[1], reverse=True)}
print(class_names)

# for i in range(0, 10):
#     predictions = model_cnn.predict(tf.expand_dims(x_test_padded[i], 0))
#     classes = np.argmax(predictions, axis=1)
#     print("real: " + str(y_test[i]) + " ,  label : " + str(classes[0])
#           + " ,  predict: " + str((predictions[0] * 100).astype(int)))
