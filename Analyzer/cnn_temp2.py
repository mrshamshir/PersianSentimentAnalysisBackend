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
from os import path

import codecs

# Keras
from keras import optimizers
from keras.models import Model, Sequential, save_model, load_model
from keras.layers import Dense, Input, Embedding, Dropout
from keras.layers import GlobalMaxPool1D, MaxPooling1D, GlobalMaxPooling1D

from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.models import model_from_json

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

model_cnn = Sequential()
model_cnn.add(Embedding(vocab_size, 300, input_length=max_length))
model_cnn.add(Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Conv1D(filters=64, kernel_size=8, activation='relu', padding='same'))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Conv1D(filters=64, kernel_size=16, activation='relu', padding='same'))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dropout(0.1))
model_cnn.add(Dense(500, activation="sigmoid"))
model_cnn.add(Dense(5, activation='softmax'))

model_cnn.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[categorical_accuracy])

model_cnn.summary()
batch_size_cnn = 64
epochs_cnn = 2

# # Train model

# Evaluate model
loss_cnn, acc_cnn = model_cnn.evaluate(x_test_padded, categorical_y_test, verbose=0)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc_cnn))

hist_cnn = model_cnn.fit(x_train_padded,
                         categorical_y_train,
                         batch_size=batch_size_cnn,
                         epochs=epochs_cnn,
                         validation_data=(x_test_padded, categorical_y_test),
                         shuffle=True,
                         # callbacks=[cp_callback]  # Pass callback to training
                         )

print("Saved model to disk")
model_cnn.save('my_model_weights.h5')
model_cnn.save_weights('model_w.h5')

# Evaluate model
loss_cnn2, acc_cnn = model_cnn.evaluate(x_test_padded, categorical_y_test, verbose=0)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc_cnn))

plot_model(model_cnn, to_file='multiclass-cnn.png')