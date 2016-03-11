import numpy as np
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer
from .base import MergeLayer


__all__ = [
    "DenseLayer",
    "NINLayer",
    "AttentionLayer", 
    "EmbeddingChangeLayer"
]


class DenseLayer(Layer):
    """
    lasagne.layers.DenseLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A fully connected layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


class NINLayer(Layer):
    """
    lasagne.layers.NINLayer(incoming, num_units, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    Network-in-network layer.
    Like DenseLayer, but broadcasting across all trailing dimensions beyond the
    2nd.  This results in a convolution operation with filter size 1 on all
    trailing dimensions.  Any number of trailing dimensions is supported,
    so NINLayer can be used to implement 1D, 2D, 3D, ... convolutions.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    untie_biases : bool
        If false the network has a single bias vector similar to a dense
        layer. If true a separate bias vector is used for each trailing
        dimension beyond the 2nd.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``,
        where ``num_inputs`` is the size of the second dimension of the input.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)`` for ``untie_biases=False``, and
        a tensor of shape ``(num_units, input_shape[2], ..., input_shape[-1])``
        for ``untie_biases=True``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, NINLayer
    >>> l_in = InputLayer((100, 20, 10, 3))
    >>> l1 = NINLayer(l_in, num_units=5)

    References
    ----------
    .. [1] Lin, Min, Qiang Chen, and Shuicheng Yan (2013):
           Network in network. arXiv preprint arXiv:1312.4400.
    """
    def __init__(self, incoming, num_units, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(NINLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        self.untie_biases = untie_biases

        num_input_channels = self.input_shape[1]

        self.W = self.add_param(W, (num_input_channels, num_units), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_units,) + self.output_shape[2:]
            else:
                biases_shape = (num_units,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units) + input_shape[2:]

    def get_output_for(self, input, **kwargs):
        # cf * bc01... = fb01...
        out_r = T.tensordot(self.W, input, axes=[[0], [1]])
        # input dims to broadcast over
        remaining_dims = range(2, input.ndim)
        # bf01...
        out = out_r.dimshuffle(1, 0, *remaining_dims)

        if self.b is None:
            activation = out
        else:
            if self.untie_biases:
                # no broadcast
                remaining_dims_biases = range(1, input.ndim - 1)
            else:
                remaining_dims_biases = ['x'] * (input.ndim - 2)  # broadcast
            b_shuffled = self.b.dimshuffle('x', 0, *remaining_dims_biases)
            activation = out + b_shuffled

        return self.nonlinearity(activation)



from .recurrent import LSTMLayer
from .recurrent import LSTMLayer_withCellOut
from .noise import DropoutLayer
class AttentionLayer(MergeLayer):
    """
    Implement the attention mechanism as described in Rocktaschel's paper
    """


    def __init__(self, incomings, num_units, W_y=init.GlorotUniform(), W_h=init.GlorotUniform(), w=init.GlorotUniform(), W_p=init.GlorotUniform(), W_x=init.GlorotUniform(), mask_input=None, **kwargs):

        if len(incomings) != 2:
            raise ValueError(
                "Input to attentionLayer must be two LSTM layers.")
        self.input_layer_first = incomings[0]
        self.input_layer_second = incomings[1]
        self.num_units = num_units    
        if not isinstance(self.input_layer_first, LSTMLayer) and not isinstance (self.input_layer_first, LSTMLayer_withCellOut) and not isinstance (self.input_layer_first, DropoutLayer):
            raise ValueError(
                "First element of input list must be an instance of LSTMLayer_withOutput.")

        if not isinstance(self.input_layer_second, LSTMLayer) and not isinstance(self.input_layer_first, DropoutLayer):
            raise ValueError(
                "Second element of input list must be an instance of LSTMLayer.")


        """
        if not (self.input_layer_first.num_units == self.input_layer_second.num_units and self.input_layer_first.num_units == self.num_units):
            raise ValueError(
                "Hidden size mismatch.")

        if not (self.input_layer_first.only_return_final == False):
            raise ValueError(
                "Input LSTMLayer_withOutput must not only return final.")
        """
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        
        # Initialize parent layer
        super(AttentionLayer, self).__init__(tuple(incomings), **kwargs)


        self.W_y = self.add_param(W_y, (num_units, num_units), name="W_y")
        self.W_h = self.add_param(W_h, (num_units, num_units), name="W_h")
        self.w = self.add_param(w, tuple([1, num_units]), name="w")
        self.W_p = self.add_param(W_p, (num_units, num_units), name="W_p")
        self.W_x = self.add_param(W_x, (num_units, num_units), name="W_x")

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return (input_shape[0], self.num_units)


    def get_output_for(self, inputs, **kwargs):
        sequence_length = inputs[0].shape[1]/2
        input_first = inputs[0][(slice(None),) + (slice(0, sequence_length),)]
        input_second = inputs[1]


        mask = inputs[self.mask_incoming_index]

        if input_second.ndim == 3:
            input_second = input_second[(slice(None), -1)]

        M = nonlinearities.tanh(T.dot(input_first, self.W_y) + T.dot(input_second.dimshuffle(0, 'x', 1), self.W_h))
        # M.shape = N * L * k
        alpha = nonlinearities.softmax(T.dot(M, self.w.T).reshape((inputs[0].shape[0], sequence_length)))
        # alpha.shape = N * L
        alpha = alpha * mask
        r = T.batched_dot(alpha, input_first)
        # r.shape = N * k
        h_star = nonlinearities.tanh(T.dot(r, self.W_p) + T.dot(input_second, self.W_x))
        return h_star


class EmbeddingChangeLayer(Layer):
    """
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.tanh,
                 **kwargs):
        super(EmbeddingChangeLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[-1]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, **kwargs):
        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x','x', 0)
        return self.nonlinearity(activation)
