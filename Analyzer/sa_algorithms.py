# General
import numpy as np
import pandas as pd
import os

# sklearn
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import _pickle as cPickle
from hazm import *
from collections import Counter
from sklearn.model_selection import train_test_split

# Keras

from keras.layers import Dense, Embedding, Dropout
from keras.layers import MaxPooling1D, GlobalMaxPooling1D

from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_table(label, reset_index=False):
    path = 'Dataset/' + str(label) + '.csv'
    data = pd.read_csv(path, index_col=None, header=None, encoding="utf-8")
    if reset_index:
        data = data.sample(frac=1).reset_index(drop=True)
    x_data = data[0]
    y_data = data[1]

    cnt = Counter(y_data)
    cnt = dict(cnt)
    class_names = {k: v for k, v in sorted(cnt.items(), key=lambda item: item[1], reverse=True)}
    print(str(label) + ': ' + str(class_names))
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    return x_data, y_data


def split_dataset(x, y, size):
    xr, xt, yr, yt = train_test_split(
        x, y, test_size=size, shuffle=True)
    cnt = Counter(yr)
    cnt = dict(cnt)
    print('new Train: ' + str(cnt))

    cnt = Counter(yt)
    cnt = dict(cnt)
    print('new Test: ' + str(cnt))
    return xr, xt, yr, yt


# declare test size and train size for split
test_size = 0.25
test_num = 10

# Import & Analyze Dataset

x, y = get_table('new_org_test')
x_train, x_test, y_train, y_test = split_dataset(x, y, test_size)

with open('dataset/x_test.pkl', 'wb') as fid:
    cPickle.dump(x_test, fid)

with open('dataset/y_test.pkl', 'wb') as fid:
    cPickle.dump(y_test, fid)

with open('dataset/x_train.pkl', 'wb') as fid:
    cPickle.dump(x_train, fid)

with open('dataset/y_train.pkl', 'wb') as fid:
    cPickle.dump(y_train, fid)

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

with open('dataset/test_docs.pkl', 'wb') as fid:
    cPickle.dump(test_docs, fid)

with open('dataset/train_docs.pkl', 'wb') as fid:
    cPickle.dump(train_docs, fid)

# Machine Learning Algorithms

# When building the vocabulary ignore terms that have a document frequency strictly lower than
# the given threshold. This value is also called cut-off in the literature.
min_df = 1

num_words = 2000

# Create the tokenizer
tokenizer = Tokenizer(num_words=num_words)

# fFt the tokenizer on the training documents
tokenizer.fit_on_texts(train_docs)

# save fitted tokenizer

with open('model/tokenizer' + '_' + str(int(test_size * 100)) + '_' + str(test_num) + '.pkl', 'wb') as fid:
    cPickle.dump(tokenizer, fid)

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


## Naive Bayes


# (Multinomial) Naive Bayes Model

def nb_trainer():
    naive_bayes = Pipeline([('vect', CountVectorizer(tokenizer=word_tokenize,
                                                     analyzer='word', ngram_range=(1, 2), min_df=min_df,
                                                     lowercase=False)),
                            ('tfidf', TfidfTransformer(sublinear_tf=True)),
                            ('clf', MultinomialNB())])
    naive_bayes = naive_bayes.fit(x_train, y_train)
    naive_score = naive_bayes.score(x_test, y_test)
    print('Naive Bayes Model: ', naive_score)
    predict_nb = naive_bayes.predict(x_test)
    return naive_bayes, naive_score, predict_nb


naive_bayes, naive_score, predict_nb = nb_trainer()

# save the classifier

with open('model/NB_classifier' + '_' + str(int(test_size * 100)) + '_' + str(test_num) + '.pkl', 'wb') as fid:
    cPickle.dump(naive_bayes, fid)


## Support Vector Machine


# Linear Support Vector Machine Model

def svm_trainer():
    svm = Pipeline([('vect', CountVectorizer(tokenizer=word_tokenize,
                                             analyzer='word', ngram_range=(1, 2),
                                             min_df=min_df, lowercase=False)),
                    ('tfidf', TfidfTransformer(sublinear_tf=True)),
                    ('clf-svm', LinearSVC(loss='hinge', penalty='l2',
                                          max_iter=1000))])

    svm = svm.fit(x_train, y_train)
    linear_svc_score = svm.score(x_test, y_test)
    print('Linear SVC Model: ', linear_svc_score)
    predict_svm = svm.predict(x_test)
    return svm, linear_svc_score, predict_svm


svm, linear_svc_score, predict_svm = svm_trainer()

# save the classifier
with open('model/SVM_classifier' + '_' + str(int(test_size * 100)) + '_' + str(test_num) + '.pkl', 'wb') as fid:
    cPickle.dump(svm, fid)

# test and get result table for one single data for SVM


f1_NB = f1_score(y_test, predict_nb, average='weighted')
print("F1 score of NB model:" + str(f1_NB))

f1_SVM = f1_score(y_test, predict_svm, average='weighted')
print("F1 score of SVM model:" + str(f1_SVM))

## CNN

num_words = 2000


def cnn_trainer():
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

    # model_cnn.summary()
    print("Total params: " + str(model_cnn.count_params()))

    return model_cnn


model_cnn = cnn_trainer()
batch_size_cnn = 64
epochs_cnn = 8

# Evaluate model
loss_cnn, acc_cnn = model_cnn.evaluate(x_test_padded, categorical_y_test, verbose=0)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc_cnn))

# # Train model
hist_cnn = model_cnn.fit(x_train_padded,
                         categorical_y_train,
                         batch_size=batch_size_cnn,
                         epochs=epochs_cnn,
                         validation_data=(x_test_padded, categorical_y_test),
                         shuffle=True,
                         verbose=0
                         )

# save model
model_cnn.save('model/CNN_classifier' + '_' + str(int(test_size * 100)) + '_' + str(test_num) + '.h5')

# Evaluate model
loss_cnn2, acc_cnn = model_cnn.evaluate(x_test_padded, categorical_y_test, verbose=0)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc_cnn))
