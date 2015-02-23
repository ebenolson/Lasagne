import numpy as np
import theano
import theano.tensor as T
from .. import init
from .. import utils
from .. import nonlinearities

from .base import Layer


__all__ = [
    "PReLuLayer"
]


class PReLuLayer(Layer):

    """
    http://arxiv.org/pdf/1502.01852v1.pdf
    """

    def __init__(self, input_layer,
                 alpha=init.Uniform([0.95, 1.05]),
                 nonlinearity=nonlinearities.lrelu,
                 epsilon=0.001,
                 **kwargs):
        super(PReLuLayer, self).__init__(input_layer, **kwargs)
        self.epsilon = epsilon
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        input_shape = input_layer.get_output_shape()

        if len(input_shape) == 2:       # in case of dense layer
            self.axis = (0)
            param_shape = (input_shape[-1])
            self.alpha = self.create_param(alpha, param_shape)
        elif len(input_shape) == 4:     # in case of conv2d layer
            self.axis = (0, 2, 3)
            param_shape = (input_shape[1], 1, 1)

            # it has to be made broadcastable on the first axis
            self.alpha = theano.shared(utils.floatX(alpha(param_shape)),
                                       broadcastable=(False, True, True),
                                       borrow=True)
        else:
            raise NotImplementedError

    def get_params(self):
        return [self.alpha,]

    def get_state(self):
        return [self.alpha,]

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, *args, **kwargs):
        return self.nonlinearity(input, self.alpha)