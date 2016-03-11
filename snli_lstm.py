from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys
import getopt

# Lasagne seed for reproducibility
lasagne.random.set_rng(np.random.RandomState(1))
# Length of word vector
WORD_VECTOR_SIZE = 300
# Number of labels
NUM_LABELS = 4
# Size of embeddings and hidden unit
K_HIDDEN = 159
# Learning rate
LEARNING_RATE = 0.01
# Dropout rate
DROPOUT_RATE = 0.3
# Regularization strength
REGU = 1e-3
# All gradients above this will be clipped
GRAD_CLIP = 100
# Number of epochs to train the net
NUM_EPOCHS = 50
# Batch Size
BATCH_SIZE = 1024

# Load data
from read_snli import *
data_train, data_val, data_test = load_dataset()
# Clip size to test algorithm
data_train = data_train
data_val = data_val
data_test = data_test[0:200]
# Number of training samples and validation samples
TRAIN_SIZE = len(data_train)
VAL_SIZE = len(data_val)
# Maximum length of premise sentence
MAX_LENGTH_PREM = 0
# Maximum length of hypothesis sentence
MAX_LENGTH_HYPO = 0
# Debug: Remove unknown words in the data
from util import remove_unknown_words
# remove_unknown_words(data_train)
remove_unknown_words(data_val)

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
vocab.append('::')
MAX_LENGTH_HYPO += 1
VOCAB_SIZE = len(vocab)
word_to_ix = {word:i for i, word in enumerate(vocab)}
ix_to_word = {i:word for i, word in enumerate(vocab)}


# Build initial word_vector
from util import get_initwv_and_mask
print("Building initial word vector")
word_vector_init, embedding_w_mask = get_initwv_and_mask(vocab)
# embedding_w_mask = np.zeros((VOCAB_SIZE, WORD_VECTOR_SIZE), dtype='float32')

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
        X_prem_mask[i, 0:len(X_prem[i])] = 1
        X_prem[i] = np.pad(X_prem[i], [(0, MAX_LENGTH_PREM - len(X_prem[i]))], 'constant')

    for i in range(batch_size):
        X_hypo[i].insert(0, VOCAB_SIZE - 1)
        X_hypo_mask[i, 0:len(X_hypo[i])] = 1
        X_hypo[i] = np.pad(X_hypo[i], [(0, MAX_LENGTH_HYPO - len(X_hypo[i]))], 'constant')

    X_prem = np.asarray(X_prem, dtype='float32')
    X_hypo = np.asarray(X_hypo, dtype='float32')

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
    
    l_in_prem = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH_PREM), input_var=input_var_prem)
    # Mask layer for premise
    l_mask_prem = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH_PREM))
    # Input layer for hypothesis
    l_in_hypo = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH_HYPO), input_var=input_var_hypo)
    # Mask layer for hypothesis
    l_mask_hypo = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH_HYPO))
    # Word embedding layers
    l_in_prem_hypo = lasagne.layers.ConcatLayer([l_in_prem, l_in_hypo], axis=1)
    l_in_embedding = lasagne.layers.EmbeddingLayer(l_in_prem_hypo, 
        VOCAB_SIZE, WORD_VECTOR_SIZE, W=word_vector_init, name='EmbeddingLayer')
    # Adding this linear layer didn't increase the accuracy, so I comment it out
    # l_in_linear = lasagne.layers.EmbeddingChangeLayer(l_in_embedding, K_HIDDEN, nonlinearity=lasagne.nonlinearities.linear)
    l_in_embedding_dropout = lasagne.layers.DropoutLayer(l_in_embedding, p=DROPOUT_RATE, rescale=True)
    l_in_prem_embedding = lasagne.layers.SliceLayer(l_in_embedding_dropout, 
        slice(0, MAX_LENGTH_PREM), axis=1)
    l_in_hypo_embedding = lasagne.layers.SliceLayer(l_in_embedding,
        slice(MAX_LENGTH_PREM, MAX_LENGTH_PREM + MAX_LENGTH_HYPO), axis=1)
    # LSTM layer for premise
    l_lstm_prem = lasagne.layers.LSTMLayer_withCellOut(l_in_prem_embedding, K_HIDDEN, 
        peepholes=False, grad_clipping=GRAD_CLIP, 
        nonlinearity=lasagne.nonlinearities.tanh, 
        mask_input=l_mask_prem, only_return_final=False)
    l_lstm_prem_dropout = lasagne.layers.DropoutLayer(l_lstm_prem, p=DROPOUT_RATE, rescale=True)
    # The slicelayer extracts the cell output of the premise sentence
    l_lstm_prem_out = lasagne.layers.SliceLayer(l_lstm_prem, -1, axis=1)
    # LSTM layer for hypothesis
    # LSTM for premise and LSTM for hypothesis have different parameters
    l_lstm_hypo = lasagne.layers.LSTMLayer(l_in_hypo_embedding, K_HIDDEN, 
        peepholes=False, grad_clipping=GRAD_CLIP, 
        nonlinearity=lasagne.nonlinearities.tanh, 
        cell_init=l_lstm_prem_out, mask_input=l_mask_hypo)
    l_lstm_hypo_dropout = lasagne.layers.DropoutLayer(l_lstm_hypo, p=DROPOUT_RATE, rescale=True)
    # Isolate the last hidden unit output
    l_hypo_out = lasagne.layers.SliceLayer(l_lstm_hypo, -1, axis=1)
    # Attention layer
    l_attention = lasagne.layers.AttentionLayer([l_lstm_prem_dropout, l_lstm_hypo_dropout], K_HIDDEN, mask_input=l_mask_prem)
    # A softmax layer create probability distribution of the prediction
    l_out = lasagne.layers.DenseLayer(l_attention, num_units=NUM_LABELS,
        W=lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    # The output of the net
    network_output_train = lasagne.layers.get_output(l_out, deterministic=False)
    network_output_test = lasagne.layers.get_output(l_out, deterministic=True)

    # Theano tensor for the targets
    target_values = T.ivector('target_output')

    # The loss function is calculated as the mean of the cross-entropy
    cost = lasagne.objectives.categorical_crossentropy(network_output_train, target_values).mean()
    from lasagne.regularization import l2, regularize_layer_params
    l2_penalty = regularize_layer_params(l_out, l2) * REGU
    cost = cost + l2_penalty
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute ADAM updates for training
    print("Computing updates ...")
    # updates = lasagne.updates.adam(cost, all_params, learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-08)
    updates = lasagne.updates.adam(cost, all_params, masks=[('EmbeddingLayer.W', embedding_w_mask)], learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-08)

    """
    # Test
    test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_values).mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                    dtype=theano.config.floatX)
    """

    # Theano functions for training and computing cost
    train_acc = T.mean(T.eq(T.argmax(network_output_test, axis=1), target_values), dtype=theano.config.floatX)
    print("Compiling functions ...")
    train = theano.function([l_in_prem.input_var, l_mask_prem.input_var, l_in_hypo.input_var, l_mask_hypo.input_var, target_values], [cost, train_acc], updates=updates, allow_input_downcast=True)

    # Theano function computing the validation loss and accuracy
    val_acc = T.mean(T.eq(T.argmax(network_output_test, axis=1), target_values), dtype=theano.config.floatX)
    validate = theano.function([l_in_prem.input_var, l_mask_prem.input_var, l_in_hypo.input_var, l_mask_hypo.input_var, target_values], [cost, val_acc], allow_input_downcast=True)

    print("Training ...")
    print('Regularization strength: ', REGU)
    print('Learning rate: ', LEARNING_RATE)
    print('Dropout rate: ', DROPOUT_RATE)
    print('Hidden size: ', K_HIDDEN)
    sys.stdout.flush()
    try:
        for epoch in range(NUM_EPOCHS):
            n = 0
            avg_cost = 0.0
            count = 0
            sub_epoch = 0
            train_acc = 0
            while n < TRAIN_SIZE:
                X_prem, X_prem_mask, X_hypo, X_hypo_mask, y = get_batch_data(n, data_train)
                err, acc = train(X_prem, X_prem_mask, X_hypo, X_hypo_mask, y)
                avg_cost += err
                train_acc += acc
                n += BATCH_SIZE
                count += 1

                if (n / BATCH_SIZE) % (TRAIN_SIZE / BATCH_SIZE / 5) == 0:
                    sub_epoch += 1
                    avg_cost /= count
                    print("Sub epoch {} average loss = {}, accuracy = {}".format(sub_epoch, avg_cost, train_acc / count * 100))
                    avg_cost = 0
                    count = 0
                    train_acc = 0


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
                    sys.stdout.flush()
            

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hr:l:d:k:', ['regu=', 'learning_rate=', 'dropout_rate=', 'k_hidden='])
    except getopt.GetoptError:
        print('argtest.py -r <regu strength> -l <learning rate> -d <dropout rate> -k <hidden size>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('argtest.py -r <regu strength> -l <learning rate> -d <dropout rate> -k <hidden size>')
            sys.exit()
        elif opt in ('-r', '--regu'):
            REGU = float(arg)
        elif opt in ('-l', '--learning_rate'):
            LEARNING_RATE = float(arg)
        elif opt in ('-d', '--dropout_rate'):
            DROPOUT_RATE = float(arg)
        elif opt in ('-k', '--k_hidden'):
            K_HIDDEN = int(arg)
    main()
