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

# Helper function to get a batch of data for each training update or each validation calculation
from util import *
def get_batch_data(begin, data):
    total_size = len(data)
    end = min(begin + BATCH_SIZE, total_size)
    return get_input_matrices(data[begin:end])


def main():

    print("Building network ...")
    # Note in Rocktaschel's paper he first used a linear layer to transform wordvector
    # into vector of size K_HIDDEN. I'm assuming that this is equivalent to update W. 
    # Input layer for premise
    l_in_prem = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None, WORD_VECTOR_SIZE))
    # Mask layer for premise
    l_mask_prem = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None))
    # Input layer for hypothesis
    l_in_hypo = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None, WORD_VECTOR_SIZE))
    # Mask layer for hypothesis
    l_mask_hypo = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None))

    # LSTM layer for premise
    l_lstm_prem = lasagne.layers.LSTMLayer_withCellOut(l_in_prem, K_HIDDEN, 
        peepholes=False, grad_clipping=GRAD_CLIP, 
        nonlinearity=lasagne.nonlinearities.tanh, 
        mask_input=l_mask_prem, only_return_final=False)


    # The slicelayer extracts the cell output of the premise sentence
    l_lstm_prem_out = lasagne.layers.SliceLayer(l_lstm_prem, -1, axis=1)


    # LSTM layer for hypothesis
    # LSTM for premise and LSTM for hypothesis have different parameters
    l_lstm_hypo = lasagne.layers.LSTMLayer(l_in_hypo, K_HIDDEN, peepholes=False,
        grad_clipping=GRAD_CLIP, nonlinearity=lasagne.nonlinearities.tanh, 
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
    updates = lasagne.updates.adam(cost, all_params, learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-08)

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

            avg_cost /= TRAIN_SIZE

            print("Epoch {} average loss = {}".format(epoch, avg_cost))
            
            # Calculate validation accuracy
            n = 0
            val_err = 0
            val_acc = 0
            val_batches = 0
            while n < VAL_SIZE:
                X_prem, X_prem_mask, X_hypo, X_hypo_mask, y = get_batch_data(n, data_val)
                err, acc = validate(X_prem, X_prem_mask, X_hypo, X_hypo_mask, y)
                val_err += err
                val_acc += acc
                val_batches += 1
                n += BATCH_SIZE
                
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
