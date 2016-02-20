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
K_HIDDEN = 64

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
data_train, data_dev, data_test = get_data_processed()

X_train_prem, X_train_hypo, y_train = data_train
X_val_prem, X_val_hypo, y_val = data_dev

# Maximum length of premise sentence
MAX_LENGTH_PREM = max([len(entry) for entry in X_train_prem])
MAX_LENGTH_VAL_PREM = max([len(entry) for entry in X_val_prem])

# Maximum length of hypothesis sentence
MAX_LENGTH_HYPO = max([len(entry) for entry in X_train_hypo])
MAX_LENGTH_VAL_HYPO = max([len(entry) for entry in X_val_hypo])

# Number of training samples
TRAIN_SIZE = len(X_train_prem)

# Number of validation samples
VAL_SIZE = len(X_val_prem)

X_prem_mask = np.zeros((TRAIN_SIZE, MAX_LENGTH_PREM))
X_hypo_mask = np.zeros((TRAIN_SIZE, MAX_LENGTH_HYPO))

X_val_prem_mask = np.zeros((VAL_SIZE, MAX_LENGTH_VAL_PREM))
X_val_hypo_mask = np.zeros((VAL_SIZE, MAX_LENGTH_VAL_HYPO))

for i in range(TRAIN_SIZE):
    X_prem_mask[i, :len(X_train_prem[i])] = 1
    X_train_prem[i] = np.pad(X_train_prem[i], [(0, MAX_LENGTH_PREM - len(X_train_prem[i])), (0, 0)], 'constant')
for i in range(TRAIN_SIZE):
    X_hypo_mask[i, :len(X_train_hypo[i])] = 1
    if (X_train_hypo[i].shape[0] == 0):
        X_train_hypo[i] = X_train_hypo[i-1]
    X_train_hypo[i] = np.pad(X_train_hypo[i], [(0, MAX_LENGTH_HYPO - len(X_train_hypo[i])), (0, 0)], 'constant')

for i in range(VAL_SIZE):
    X_val_prem_mask[i, :len(X_val_prem[i])] = 1
    X_val_prem[i] = np.pad(X_val_prem[i], [(0, MAX_LENGTH_VAL_PREM - len(X_val_prem[i])), (0, 0)], 'constant')
for i in range(VAL_SIZE):
    X_val_hypo_mask[i, :len(X_val_hypo[i])] = 1
    X_val_hypo[i] = np.pad(X_val_hypo[i], [(0, MAX_LENGTH_VAL_HYPO - len(X_val_hypo[i])), (0, 0)], 'constant')
    
X_train_prem = np.asarray(X_train_prem)
X_train_hypo = np.asarray(X_train_hypo)


X_val_prem = np.asarray(X_val_prem)
X_val_hypo = np.asarray(X_val_hypo)

# Get a batch of data for each training update
def get_batch_data(begin):
    end = min(begin + BATCH_SIZE, TRAIN_SIZE)
    return X_train_prem[begin:end], X_prem_mask[begin:end], X_train_hypo[begin:end], X_hypo_mask[begin:end], y_train[begin:end]


def get_batch_data_val(begin):
    end = min(begin + BATCH_SIZE, VAL_SIZE)
    return X_val_prem[begin:end], X_val_prem_mask[begin:end], X_val_hypo[begin:end], X_val_hypo_mask[begin:end], y_val[begin:end]

def main():
    print("Building network ...")

    # Note in Rocktaschel's paper he first used a linear layer to transform wordvector
    # into vector of size K_HIDDEN. I'm assuming that this is equivalent to update W. 
    # Input layer for premise
    l_in_prem_input = lasagne.layers.InputLayer(shape=(BATCH_SIZE, MAX_LENGTH_PREM, WORD_VECTOR_SIZE))
    l_in_prem = lasagne.layers.EmbeddingChangeLayer(l_in_prem_input, K_HIDDEN)
    # Mask layer for premise
    l_mask_prem = lasagne.layers.InputLayer(shape=(BATCH_SIZE, MAX_LENGTH_PREM))
    # Input layer for hypothesis
    l_in_hypo_input = lasagne.layers.InputLayer(shape=(BATCH_SIZE, MAX_LENGTH_HYPO, WORD_VECTOR_SIZE))
    l_in_hypo = lasagne.layers.EmbeddingChangeLayer(l_in_hypo_input, K_HIDDEN)
    # Mask layer for hypothesis
    l_mask_hypo = lasagne.layers.InputLayer(shape=(BATCH_SIZE, MAX_LENGTH_HYPO))

    # LSTM layer for premise
    l_lstm_prem = lasagne.layers.LSTMLayer_withCellOut(l_in_prem, K_HIDDEN, 
        peepholes=False, grad_clipping=GRAD_CLIP, 
        nonlinearity=lasagne.nonlinearities.tanh, 
        mask_input=l_mask_prem, only_return_final=False)

    l_lstm_prem_out = lasagne.layers.SliceLayer(l_lstm_prem, -1, axis=1)


    # The slicelayer extracts the cell output of the premise sentence
    # l_inter_cell = lasagne.layers.SliceLayer(l_lstm_prem, -1, 1)


    # LSTM layer for hypothesis
    # LSTM for premise and LSTM for hypothesis have different parameters
    l_lstm_hypo = lasagne.layers.LSTMLayer(l_in_hypo, K_HIDDEN, peepholes=False,
        grad_clipping=GRAD_CLIP, nonlinearity=lasagne.nonlinearities.tanh, 
        cell_init=l_lstm_prem_out, mask_input=l_mask_hypo)

    # Isolate the last hidden unit output
    l_hypo_out = lasagne.layers.SliceLayer(l_lstm_hypo, -1, 1)

    l_attention = lasagne.layers.AttentionLayer([l_lstm_prem, l_lstm_hypo], K_HIDDEN, mask_input=l_mask_prem)

    # A softmax layer create probability distribution of the prediction
    l_out = lasagne.layers.DenseLayer(l_attention, num_units=NUM_LABELS,
        W=lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    # Theano tensor for the targets
    target_values = T.ivector('target_output')

    # The output of the net
    network_output = lasagne.layers.get_output(l_out)

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
    train = theano.function([l_in_prem_input.input_var, l_mask_prem.input_var, l_in_hypo_input.input_var, l_mask_hypo.input_var, target_values], cost, updates=updates, allow_input_downcast=True)

    #compute_cost = theano.function([l_in_prem.input_var, l_mask_prem.input_var, l_in_hypo.input_var, l_mask_hypo.input_var, target_values], cost)

    """
    # Theano function computing the validation loss and accuracy
    validate = theano.function([input_var, target_var], 
    """
    val_acc = T.mean(T.eq(T.argmax(network_output, axis=1), target_values), dtype=theano.config.floatX)

    validate = theano.function([l_in_prem_input.input_var, l_mask_prem.input_var, l_in_hypo_input.input_var, l_mask_hypo.input_var, target_values], [cost, val_acc])

    print("Training ...")
    try:
        for epoch in range(NUM_EPOCHS):
            n = 0
            avg_cost = 0.0
            while n < TRAIN_SIZE:
                X_prem, X_prem_mask, X_hypo, X_hypo_mask, y = get_batch_data(n)
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
                X_prem, X_prem_mask, X_hypo, X_hypo_mask, y = get_batch_data_val(n)
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
