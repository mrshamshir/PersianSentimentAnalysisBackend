import _pickle as cPickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
print(str(BASE_DIR))

# load NB_classifier again
with open(BASE_DIR / 'model/NB_classifier.pkl', 'rb') as fid:
    nb = cPickle.load(fid)

# load SVM_classifier again
with open(BASE_DIR / 'model/SVM_classifier.pkl', 'rb') as fid:
    svm = cPickle.load(fid)

# load SGD_classifier again
with open(BASE_DIR / 'model/SGD_classifier.pkl', 'rb') as fid:
    sgd = cPickle.load(fid)


def get_classes(comment):
    result_table = nb.predict_proba([comment])[0]
    result_value = nb.predict([comment])

    keys = ['Furious', 'Angry', 'Neutral', 'Happy', 'Delighted']
    dictionary = dict(zip(keys, result_table*100))

    return dictionary, result_value
