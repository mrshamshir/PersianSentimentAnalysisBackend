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

# Preprocessing
## from StopWords_persian import stopwords_output

from hazm import *

# Import & Analyze Dataset
test = pd.read_csv('Dataset/test.csv', index_col=None, header=None, encoding="utf-8")

x_test = test[0]
y_test = test[1]

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

original = pd.read_csv('Dataset/original.csv', index_col=None, header=None, encoding="utf-8")
balanced = pd.read_csv('Dataset/balanced.csv', index_col=None, header=None, encoding="utf-8")
translation = pd.read_csv('dataset/translation.csv', index_col=None, header=None, encoding="utf-8")

selected_dataset = balanced

selected_dataset = selected_dataset.sample(frac=1).reset_index(drop=True)

x_train = selected_dataset[0]
y_train = selected_dataset[1]

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

# Make stop word set

file = pd.read_csv('dataset/per_sw.csv', sep="\n", encoding="utf-8")
stop_set = set(file.values.flatten())

# Machine Learning Algorithms

# When building the vocabulary ignore terms that have a document frequency strictly lower than
# the given threshold. This value is also called cut-off in the literature.
min_df = 1


# Tokenize function used in Vectorizer
def tokenize(text):
    return word_tokenize(text)


## Naive Bayes


# (Multinomial) Naive Bayes Model
naive_bayes = Pipeline([('vect', CountVectorizer(tokenizer=tokenize,
                                                 analyzer='word', ngram_range=(1, 2), min_df=min_df, lowercase=False)),
                        ('tfidf', TfidfTransformer(sublinear_tf=True)),
                        ('clf', MultinomialNB())])
naive_bayes = naive_bayes.fit(x_train, y_train)
naive_score = naive_bayes.score(x_test, y_test)
print('Naive Bayes Model: ', naive_score)
predict_nb = naive_bayes.predict(x_test)

# test and get result table for one single data for NB
# x = 'حساسیت لمسی خود نمایشگر هم کاملا عالیست.'
# y = 'سرعت اجرا بسيار بالا است و مصرف باتري نيز مناسب است.'
# z = 'فارسيش هم عاليه اين گوشي تازه اومده.'
# w = 'این به این معناست که از صفحه نمایش آن نمی‌توان انتظار تصویر چندان خوبی در مقابل صفحه نمایش‌های Super AMOLED و یا Super Clear LCD را داشت.'
# temp = naive_bayes.predict_proba([y, z, w])
#
# print(naive_bayes.classes_)
# print(naive_bayes.predict([y, z, w]))
# print(temp)


## Support Vector Machine


# Linear Support Vector Machine Model
svm = Pipeline([('vect', CountVectorizer(tokenizer=tokenize,
                                         analyzer='word', ngram_range=(1, 2),
                                         min_df=min_df, lowercase=False)),
                ('tfidf', TfidfTransformer(sublinear_tf=True)),
                ('clf-svm', LinearSVC(loss='hinge', penalty='l2',
                                      max_iter=1000))])

svm = svm.fit(x_train, y_train)
linear_svc_score = svm.score(x_test, y_test)
print('Linear SVC Model: ', linear_svc_score)
predict_svm = svm.predict(x_test)

# test and get result table for one single data for SVM

# x = svm.decision_function([x_test[0]])
# y = svm.predict([x_test[0]])
# print(x)
# print(y)

## Stochastic Gradient Descent


# SGD (Stochastic Gradient Descent) Model
sgd = Pipeline([('vect', CountVectorizer(tokenizer=tokenize,
                                         analyzer='word', ngram_range=(1, 2), min_df=min_df, lowercase=False)),
                ('tfidf', TfidfTransformer(sublinear_tf=True)),
                ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                          alpha=1e-3, max_iter=1000))])
sgd = sgd.fit(x_train, y_train)
sgd_score = sgd.score(x_test, y_test)
print('SGD Model: ', sgd_score)
predict_sgd = sgd.predict(x_test)

f1_NB = f1_score(y_test, predict_nb, average='weighted')
print("F1 score of NB model:" + str(f1_NB))

f1_SVM = f1_score(y_test, predict_svm, average='weighted')
print("F1 score of SVM model:" + str(f1_SVM))

f1_SGD = f1_score(y_test, predict_sgd, average='weighted')
print("F1 score of SGD model:" + str(f1_SGD))
