import json
import numpy as np
import os

WORDVECTORS_PATH = ["/Users/david/Documents/Stanford/SNLI/data/GoogleNews-vectors-negative300.bin", "/juicier/scr100/scr/qiaojing/snli/data/GoogleNews-vectors-negative300.bin"]

# load model to get word vector
from gensim.models import word2vec
print("Loading word vectors ...")
model = None
for path in WORDVECTORS_PATH:
    if os.path.exists(path): 
        model = word2vec.Word2Vec.load_word2vec_format(path, binary=True)
        break

if model == None:
    print("Cannot find file for word vectors");

def get_wordvector(sentence):
    sentence_wv = []
    for word in sentence.split(' '):
        try:
            sentence_wv.append(model[word.strip(",.!?:;")])
        except KeyError:
            # For unknown words, we currently just ignore them
            pass
    return np.asarray(sentence_wv)


def process(data):
    X_prem = []
    X_hypo = []
    y = []

    label_to_num = {'neutral': 0, 'entailment': 1, 'contradiction': 2, '-': 3}
    for entry in data:
        sentence1 = entry['sentence1']
        sentence2 = entry['sentence2']
        gold_label = entry['gold_label']
        # convert the first letter of sentence to lower case
        # currently not implemented
        vector1 = get_wordvector(sentence1)
        if (vector1.shape[0] == 0):
            continue
        vector2 = get_wordvector(sentence2)
        if (vector2.shape[0] == 0):
            continue
        if (gold_label == '-'):
            continue
        X_prem.append(vector1)
        X_hypo.append(vector2)
        y.append(label_to_num[gold_label]);

    return (X_prem, X_hypo, y)


def get_input_matrices(batch_data):
    X_prem, X_hypo, y = process(batch_data)
    batch_size = len(X_prem)

    # Maximum length of premise sentence
    MAX_LENGTH_PREM = max([len(entry) for entry in X_prem])
    # Maximum length of hypothesis sentence
    MAX_LENGTH_HYPO = max([len(entry) for entry in X_hypo])

    # Mask is used in Lasagne LSTM layer
    X_prem_mask = np.zeros((batch_size, MAX_LENGTH_PREM))
    X_hypo_mask = np.zeros((batch_size, MAX_LENGTH_HYPO))
    

    for i in range(batch_size):
        X_prem_mask[i, :len(X_prem[i])] = 1
        X_prem[i] = np.pad(X_prem[i], [(0, MAX_LENGTH_PREM - len(X_prem[i])), (0, 0)], 'constant')

    for i in range(batch_size):
        X_hypo_mask[i, :len(X_hypo[i])] = 1
        X_hypo[i] = np.pad(X_hypo[i], [(0, MAX_LENGTH_HYPO - len(X_hypo[i])), (0, 0)], 'constant')

    X_prem = np.asarray(X_prem)
    X_hypo = np.asarray(X_hypo)

    y = np.asarray(y)

    return X_prem, X_prem_mask, X_hypo, X_hypo_mask, y



