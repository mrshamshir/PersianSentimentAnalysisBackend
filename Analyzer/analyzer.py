import _pickle as cPickle
from pathlib import Path
import numpy as np
import pandas as pd
from hazm import *
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
print(str(BASE_DIR))


def get_nb_classifier_result(comment, classifier, test_size):
    # path = 'model/' + str(classifier) + '_classifier' + str(test_size) + ".pkl"
    path = 'model/NB_classifier_' + str(test_size) + "_1.pkl"
    with open(BASE_DIR / path, 'rb') as fid:
        nb = cPickle.load(fid)
    result_table = nb.predict_proba([comment])[0]
    result_value = nb.predict([comment])[0]
    keys = ['Furious', 'Angry', 'Neutral', 'Happy', 'Delighted']
    dictionary = dict(zip(keys, result_table * 100))
    for temp in dictionary:
        dictionary[temp] = round(dictionary[temp], 2)
    dictionary['final'] = result_value
    return dictionary


def get_svm_classifier_result(comment, classifier, test_size):
    # path = 'model/' + str(classifier) + '_classifier' + str(test_size) + ".pkl"
    path = 'model/SVM_classifier_' + str(test_size) + "_1.pkl"
    with open(BASE_DIR / path, 'rb') as fid:
        svm = cPickle.load(fid)
    result_table = svm.decision_function([comment])[0]

    result_value = svm.predict([comment])[0]
    keys = ['Furious', 'Angry', 'Neutral', 'Happy', 'Delighted']
    dictionary = dict(zip(keys, result_table))
    for temp in dictionary:
        dictionary[temp] = round(dictionary[temp], 2)
    dictionary['final'] = result_value
    return dictionary


def get_cnn_classifier_result(comment, classifier, test_size):
    # path = 'model/' + str(classifier) + '_classifier' + str(test_size) + ".h5"
    path = 'model/CNN_classifier_' + str(test_size) + "_1.h5"
    cnn = load_model(BASE_DIR / path)
    path = 'model/tokenizer_' + str(test_size) + '_1.pkl'
    with open(BASE_DIR / path, 'rb') as handle:
        tokenizer = cPickle.load(handle)

    max_length = 171
    comment = [comment, comment]

    encoded_comment = tokenizer.texts_to_sequences(comment)

    comment_padded = pad_sequences(encoded_comment, maxlen=max_length, padding='post')

    predictions = cnn.predict(tf.expand_dims(comment_padded[0], 0))
    classes = np.argmax(predictions, axis=1)
    result_value = classes - 2
    result_table = (predictions[0] * 100)

    result_table = result_table * 100
    result_table = result_table.astype(int)
    result_table = result_table / 100

    keys = ['Furious', 'Angry', 'Neutral', 'Happy', 'Delighted']
    dictionary = dict(zip(keys, result_table))

    dictionary['final'] = result_value[0]
    return dictionary


def preprocess_comment(comment):
    puncs = ['،', '.', ',', ':', ';', '"']
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()

    # Make stop word set
    file = pd.read_csv(BASE_DIR / 'dataset/per_sw.csv', sep="\n", encoding="utf-8")
    stop_set = set(file.values.flatten())

    comment = normalizer.normalize(comment)  # Normalize document using Hazm Normalizer
    tokenized = word_tokenize(comment)  # Tokenize text
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
    final_comment = ' '.join(tokens)

    return final_comment


def get_classes(comment, classifier, test_size):
    comment = preprocess_comment(comment)
    if classifier == 'NB':
        return get_nb_classifier_result(comment, classifier, test_size)
    elif classifier == 'SVM':
        return get_svm_classifier_result(comment, classifier, test_size)
    elif classifier == 'CNN':
        return get_cnn_classifier_result(comment, classifier, test_size)
    return "okok"
