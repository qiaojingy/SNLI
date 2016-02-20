import sys
import os
import json
import numpy as np
import pickle

SNLI_DIR = ["/Users/david/Documents/Stanford/SNLI/data/snli_1.0", "/scr/nlp/data/snli_1.0"]
directory = None
for path in SNLI_DIR:
    if os.path.isdir(path):
        directory = path
if directory == None:
    print("snli data not found")

def load_dataset():
    def download_data():
        print("Downloading SNLI corpus ...");
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrive
        urlretrieve("http://nlp.stanford.edu/projects/snli/snli_1.0.zip", 
            "snli_1.0.zip");

        import zipfile
        z = zipfile.ZipFile("snli_1.0.zip", 'r')
        z.extractall()

    def load_snli_data(filename):
        f = open(directory + '/' + filename)
        data = []
        while (1):
            line = f.readline()
            if not line: 
                break
            data.append(json.loads(line))
        return data

    data_train = load_snli_data('snli_1.0_train.jsonl')
    data_dev = load_snli_data('snli_1.0_dev.jsonl')
    data_test = load_snli_data('snli_1.0_test.jsonl')
    return data_train, data_dev, data_test

def get_data_processed():

    # get the wordvectors as numpy array from a sentence
    # return a numpy array of shape (len(sentence), |wordvector|)
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
            X_prem.append(vector1)
            vector2 = get_wordvector(sentence2)
            if (vector2.shape[0] == 0):
                continue
            X_hypo.append(vector2)
            y.append(label_to_num[gold_label]);

        return (X_prem, X_hypo, y)

    import os.path
    if (os.path.isfile(directory + '/train_processed.dat')):
        with open(directory + '/train_processed.dat', 'rb') as f:
            train_processed = pickle.load(f)
            f.close()
        with open(directory + '/dev_processed.dat', 'rb') as f:
            dev_processed = pickle.load(f)
            f.close()
        with open(directory + '/test_processed.dat', 'rb') as f:
            test_processed = pickle.load(f)
            f.close()

    else:
        data_train, data_dev, data_test = load_dataset()

        # load model to get word vector
        from gensim.models import word2vec
        model = word2vec.Word2Vec.load_word2vec_format(directory + '/../GoogleNews-vectors-negative300.bin', binary=True)
        
        # train_processed = X_train_prem, X_train_hypo, y_train
        train_processed = process(data_train)
        dev_processed = process(data_dev)
        test_processed = process(data_test)

        with open(directory + '/snli_1.0/train_processed.dat', 'wb') as f:
            pickle.dump(train_processed, f)
            f.close()
        with open(direcotory +'/snli_1.0/dev_processed.dat', 'wb') as f:
            pickle.dump(dev_processed, f)
            f.close()
        with open(directory + '/snli_1.0/test_processed.dat', 'wb') as f:
            pickle.dump(test_processed, f)
            f.close()

    return train_processed, dev_processed, test_processed

if __name__ == '__main__':
    get_data_processed()


    
