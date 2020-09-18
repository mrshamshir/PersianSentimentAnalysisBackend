# General
import numpy as np
import pandas as pd


# sklearn
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# Preprocessing
## from StopWords_persian import stopwords_output

from hazm import *

# Import & Analyze Dataset

# test = pd.read_csv('Dataset/test.csv', index_col=None, header=None, encoding="utf-8")

# x_test = test[0]
# y_test = test[1]

# print('Number of testing sentence: ', x_test.shape)
# print('Number of testing label: ', y_test.shape)

# x_test = np.asarray(x_test)
# y_test = np.asarray(y_test)

# original = pd.read_csv('dataset/original.csv', index_col=None, header=None, encoding="utf-8")
# balanced = pd.read_csv('dataset/balanced.csv', index_col=None, header=None, encoding="utf-8")
translation = pd.read_csv('dataset/translation.csv', index_col=None, header=None, encoding="utf-8")

selected_dataset = translation

selected_dataset = selected_dataset.sample(frac=1).reset_index(drop=True)

x_train = selected_dataset[0]
y_train = selected_dataset[1]

# print('Number of training sentence: ', x_train.shape)
# print('Number of training label: ', y_train.shape)

# Convert dataframes to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# See the data number of sentence in each category
# from collections import Counter

# cnt = Counter(y_train)
# cnt = dict(cnt)
# print(cnt)
#
# labels = list(cnt.keys())
# sizes = list(cnt.values())
# colors = ['#3fba36', '#66b3ff', '#ffcc99', '#ff9999', '#d44444']
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, colors=colors,
#         autopct='%1.1f%%', startangle=90)

# draw circle
# centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
# ax1.axis('equal')
# plt.tight_layout()
# Decomment following line if you want to save the figure
# plt.savefig('Plot/distribution.png')
# plt.show()

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
# test_docs = np.empty_like(x_test)
# for index, document in enumerate(x_test):
#     test_docs[index] = clean_doc(document)

# Machine Learning Algorithms


# Make stop word set
stop_set = stopwords_output("Persian", "set")

# When building the vocabulary ignore terms that have a document frequency strictly lower than
# the given threshold. This value is also called cut-off in the literature.
min_df = 1


# Tokenize function used in Vectorizer
def tokenize(text):
    return word_tokenize(text)


## Naive Bayes


# # (Multinomial) Naive Bayes Model
# naive_bayes = Pipeline([('vect', CountVectorizer(tokenizer=tokenize,
#                                                  analyzer='word', ngram_range=(1, 2), min_df=min_df, lowercase=False)),
#                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
#                         ('clf', MultinomialNB())])
# naive_bayes = naive_bayes.fit(x_train, y_train)
# naive_score = naive_bayes.score(x_test, y_test)
# print('Naive Bayes Model: ', naive_score)
# predict_nb = naive_bayes.predict(x_test)

## Support Vector Machine


# Linear Support Vector Machine Model
svm = Pipeline([('vect', CountVectorizer(tokenizer=tokenize,
                                         analyzer='word', ngram_range=(1, 2),
                                         min_df=min_df, lowercase=False)),
                ('tfidf', TfidfTransformer(sublinear_tf=True)),
                ('clf-svm', LinearSVC(loss='hinge', penalty='l2',
                                      max_iter=1000))])

svm = svm.fit(x_train, y_train)
# linear_svc_score = svm.score(x_test, y_test)
# print('Linear SVC Model: ', linear_svc_score)
predict_svm = svm.predict(x_test)

# ## Stochastic Gradient Descent
#
#
# # SGD (Stochastic Gradient Descent) Model
# sgd = Pipeline([('vect', CountVectorizer(tokenizer=tokenize,
#                                          analyzer='word', ngram_range=(1, 2), min_df=min_df, lowercase=False)),
#                 ('tfidf', TfidfTransformer(sublinear_tf=True)),
#                 ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
#                                           alpha=1e-3, max_iter=1000))])
# sgd = sgd.fit(x_train, y_train)
# sgd_score = sgd.score(x_test, y_test)
# print('SGD Model: ', sgd_score)
# predict_sgd = sgd.predict(x_test)


# Confusion Matrix


# from sklearn.metrics import confusion_matrix
# from sklearn.utils.multiclass import unique_labels
#
#
# def plot_confusion_matrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     print(im)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return ax

# class_names = np.array([0, 1, 2, -2, -1])
# np.set_printoptions(precision=2)

# y_test = y_test.astype(int)
# predict_nb = predict_nb.astype(int)
# predict_svm = predict_svm.astype(int)
# predict_sgd = predict_sgd.astype(int)

# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, predict_nb, classes=class_names)
# # plt.savefig('cm-nb.png')
# # Plot normalized confusion matrix
# plot_confusion_matrix(y_test, predict_nb, classes=class_names, normalize=True)
# # plt.savefig('cm-nb-normalized.png')
# plt.show()
#
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, predict_svm, classes=class_names)
# # plt.savefig('cm-svm.png')
# # Plot normalized confusion matrix
# plot_confusion_matrix(y_test, predict_svm, classes=class_names, normalize=True)
# # plt.savefig('cm-svm-normalized.png')
# plt.show()
#
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, predict_sgd, classes=class_names)
# # plt.savefig('cm-sgd.png')
# # Plot normalized confusion matrix
# plot_confusion_matrix(y_test, predict_sgd, classes=class_names, normalize=True)
# # plt.savefig('cm-sgd-normalized.png')
# plt.show()

## F1 Score


# f1_NB = f1_score(y_test, predict_nb, average='weighted')
# print("F1 score of NB model:" + str(f1_NB))
#
# f1_SVM = f1_score(y_test, predict_svm, average='weighted')
# print("F1 score of SVM model:" + str(f1_SVM))
#
# f1_SGD = f1_score(y_test, predict_sgd, average='weighted')
# print("F1 score of SGD model:" + str(f1_SGD))


