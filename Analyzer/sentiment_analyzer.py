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

# Preprocessing
## from StopWords_persian import stopwords_output

from hazm import *

# Import & Analyze Dataset

translation = pd.read_csv('dataset/translation.csv', index_col=None, header=None, encoding="utf-8")

selected_dataset = translation

selected_dataset = selected_dataset.sample(frac=1).reset_index(drop=True)

x_train = selected_dataset[0]
y_train = selected_dataset[1]
print(y_train)

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

# Machine Learning Algorithms


# Make stop word set


file = pd.read_csv('dataset/per_sw.csv', sep="\n", encoding="utf-8")
numpy_array = file.values
stop_set = set(numpy_array.flatten())

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
x = 'حساسیت لمسی خود نمایشگر هم کاملا عالیست.'
y = 'سرعت اجرا بسيار بالا است و مصرف باتري نيز مناسب است.'
z = 'فارسيش هم عاليه اين گوشي تازه اومده.'
w = 'این به این معناست که از صفحه نمایش آن نمی‌توان انتظار تصویر چندان خوبی در مقابل صفحه نمایش‌های Super AMOLED و یا Super Clear LCD را داشت.'
temp = naive_bayes.predict_proba([y, z, w])
temp2 = naive_bayes.predict_log_proba([y, z, w])

print(temp)
print(naive_bayes.predict([y, z, w]))
print(temp2)
