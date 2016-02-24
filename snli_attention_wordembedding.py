from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne

# Lasagne seed for reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Length of word vector
WORD_VECTOR_SIZE = 300

# Number of labels
NUM_LABELS = 4

# Size of embeddings and hidden unit
K_HIDDEN = 300

# Learning rate
LEARNING_RATE = 0.01

# All gradients above this will be clipped
GRAD_CLIP = 100

# Number of epochs to train the net
NUM_EPOCHS = 20

# Batch Size
BATCH_SIZE = 32

# Load data
from read_snli import *

data_train, data_val, data_test = load_dataset()

# Clip size to test algorithm
data_train = data_train[0:1000]
data_val = data_val[0:200]
data_test = data_test[0:200]

# Number of training samples and validation samples
TRAIN_SIZE = len(data_train)
VAL_SIZE = len(data_val)

# Maximum length of premise sentence
MAX_LENGTH_PREM = 0
# Maximum length of hypothesis sentence
MAX_LENGTH_HYPO = 0
# Build vocabulary
vocab = set()
for entry in data_train:
    vocab.update(w.strip(",.!?:;") for w in entry['sentence1'].split(' '))
    vocab.update(w.strip(",.!?:;") for w in entry['sentence2'].split(' '))
    MAX_LENGTH_PREM = max(MAX_LENGTH_PREM, len(entry['sentence1'].split(' ')))
    MAX_LENGTH_HYPO = max(MAX_LENGTH_HYPO, len(entry['sentence2'].split(' ')))
for entry in data_val:
    vocab.update(w.strip(",.!?:;") for w in entry['sentence1'].split(' '))
    vocab.update(w.strip(",.!?:;") for w in entry['sentence2'].split(' '))
    MAX_LENGTH_PREM = max(MAX_LENGTH_PREM, len(entry['sentence1'].split(' ')))
    MAX_LENGTH_HYPO = max(MAX_LENGTH_HYPO, len(entry['sentence2'].split(' ')))

vocab = list(vocab)
vocab.append('::');
MAX_LENGTH_HYPO += 1
VOCAB_SIZE = len(vocab)
word_to_ix = {word:i for i, word in enumerate(vocab)}
ix_to_word = {i:word for i, word in enumerate(vocab)}

print(MAX_LENGTH_PREM)
print(MAX_LENGTH_HYPO)

# Build initial word_vector
from util import word_to_vector
word_vector_init = np.asarray([word_to_vector(word) for word in vocab], dtype='float32')

# Helper function to get a batch of data for each training update or each validation calculation
def get_input_matrices_2(batch_data):
    X_prem = []
    X_hypo = []
    y = []

    label_to_num = {'neutral': 0, 'entailment': 1, 'contradiction': 2, '-': 3}
    for entry in batch_data:
        sentence1 = entry['sentence1']
        sentence2 = entry['sentence2']
        gold_label = entry['gold_label']
        # convert the first letter of sentence to lower case
        # currently not implemented
        vector1 = [word_to_ix[w.strip(",.!?:;")] for w in sentence1.split(' ')]
        vector2 = [word_to_ix[w.strip(",.!?:;")] for w in sentence2.split(' ')]
        if (len(vector1) == 0):
            continue
        if (len(vector2) == 0):
            continue
        if (gold_label == '-'):
            continue
        X_prem.append(vector1)
        X_hypo.append(vector2)
        y.append(label_to_num[gold_label]);

    batch_size = len(X_prem)


    # Mask is used in Lasagne LSTM layer
    X_prem_mask = np.zeros((batch_size, MAX_LENGTH_PREM))
    X_hypo_mask = np.zeros((batch_size, MAX_LENGTH_HYPO))
    
    for i in range(batch_size):
        X_prem_mask[i, :len(X_prem[i])] = 1
        X_prem[i] = np.pad(X_prem[i], [(0, MAX_LENGTH_PREM - len(X_prem[i]))], 'constant')

    for i in range(batch_size):
        X_hypo_mask[i, :len(X_hypo[i])] = 1
        X_hypo[i].insert(0, VOCAB_SIZE - 1)
        X_hypo[i] = np.pad(X_hypo[i], [(0, MAX_LENGTH_HYPO - len(X_hypo[i]))], 'constant')

    return (X_prem, X_prem_mask, X_hypo, X_hypo_mask, y)

def get_batch_data(begin, data):
    total_size = len(data)
    end = min(begin + BATCH_SIZE, total_size)
    return get_input_matrices_2(data[begin:end])


def main():

    print("Building network ...")
    # Note in Rocktaschel's paper he first used a linear layer to transform wordvector
    # into vector of size K_HIDDEN. I'm assuming that this is equivalent to update W. 
    # Input layer for premise
    input_var_type = T.TensorType('int32', [False] * 2)
    var_name = "input"
    input_var_prem = input_var_type(var_name)
    input_var_hypo = input_var_type(var_name)
    
    l_in_prem = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None), input_var=input_var_prem)
    # Mask layer for premise
    l_mask_prem = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None))
    # Input layer for hypothesis
    l_in_hypo = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None), input_var=input_var_hypo)
    # Mask layer for hypothesis
    l_mask_hypo = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None))

    # Word embedding layers
    l_in_prem_hypo = lasagne.layers.ConcatLayer([l_in_prem, l_in_hypo], axis=1)
    l_in_embedding = lasagne.layers.EmbeddingLayer(l_in_prem_hypo, 
        VOCAB_SIZE, WORD_VECTOR_SIZE, W=word_vector_init, name='EmbeddingLayer')
    l_in_prem_embedding = lasagne.layers.SliceLayer(l_in_embedding, 
        slice(0, MAX_LENGTH_PREM), axis=1)
    l_in_hypo_embedding = lasagne.layers.SliceLayer(l_in_embedding,
        slice(MAX_LENGTH_PREM, MAX_LENGTH_PREM + MAX_LENGTH_HYPO), axis=1)

    # LSTM layer for premise
    l_lstm_prem = lasagne.layers.LSTMLayer_withCellOut(l_in_prem_embedding, K_HIDDEN, 
        peepholes=False, grad_clipping=GRAD_CLIP, 
        nonlinearity=lasagne.nonlinearities.tanh, 
        mask_input=l_mask_prem, only_return_final=False)


    # The slicelayer extracts the cell output of the premise sentence
    l_lstm_prem_out = lasagne.layers.SliceLayer(l_lstm_prem, -1, axis=1)


    # LSTM layer for hypothesis
    # LSTM for premise and LSTM for hypothesis have different parameters
    l_lstm_hypo = lasagne.layers.LSTMLayer(l_in_hypo_embedding, K_HIDDEN, 
        peepholes=False, grad_clipping=GRAD_CLIP, 
        nonlinearity=lasagne.nonlinearities.tanh, 
        cell_init=l_lstm_prem_out, mask_input=l_mask_hypo)

    # Isolate the last hidden unit output
    l_hypo_out = lasagne.layers.SliceLayer(l_lstm_hypo, -1, 1)

    # Attention layer
    l_attention = lasagne.layers.AttentionLayer([l_lstm_prem, l_lstm_hypo], K_HIDDEN, mask_input=l_mask_prem)
    # A softmax layer create probability distribution of the prediction
    l_out = lasagne.layers.DenseLayer(l_attention, num_units=NUM_LABELS,
        W=lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    # The output of the net
    network_output = lasagne.layers.get_output(l_out)

    # Theano tensor for the targets
    target_values = T.ivector('target_output')

    # The loss function is calculated as the mean of the cross-entropy
    cost = lasagne.objectives.categorical_crossentropy(network_output, target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute ADAM updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adam(cost, all_params, masks=[('EmbeddingLayer.W', np.zeros((VOCAB_SIZE, WORD_VECTOR_SIZE), dtype='float32'))], learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-08)

    """
    # Test
    test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_values).mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                    dtype=theano.config.floatX)
    """

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in_prem.input_var, l_mask_prem.input_var, l_in_hypo.input_var, l_mask_hypo.input_var, target_values], cost, updates=updates, allow_input_downcast=True)

    # Theano function computing the validation loss and accuracy
    val_acc = T.mean(T.eq(T.argmax(network_output, axis=1), target_values), dtype=theano.config.floatX)
    validate = theano.function([l_in_prem.input_var, l_mask_prem.input_var, l_in_hypo.input_var, l_mask_hypo.input_var, target_values], [cost, val_acc], allow_input_downcast=True)

    print("Training ...")
    try:
        for epoch in range(NUM_EPOCHS):
            n = 0
            avg_cost = 0.0
            while n < TRAIN_SIZE:
                X_prem, X_prem_mask, X_hypo, X_hypo_mask, y = get_batch_data(n, data_train)
                avg_cost += train(X_prem, X_prem_mask, X_hypo, X_hypo_mask, y)
                n += BATCH_SIZE
                # Calculate validation accuracy
                m = 0
                val_err = 0
                val_acc = 0
                val_batches = 0
                while m < VAL_SIZE:
                    X_prem, X_prem_mask, X_hypo, X_hypo_mask, y = get_batch_data(m, data_val)
                    err, acc = validate(X_prem, X_prem_mask, X_hypo, X_hypo_mask, y)
                    val_err += err
                    val_acc += acc
                    val_batches += 1
                    m += BATCH_SIZE
                    
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

            avg_cost /= TRAIN_SIZE

            print("Epoch {} average loss = {}".format(epoch, avg_cost))
            

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
