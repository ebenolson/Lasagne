import numpy as np
import theano
import theano.tensor as T

from .base import Layer

__all__ = [
    "BatchNormalizationLayer",
]


class BatchNormalizationLayer(Layer):
    def __init__(self, incoming, epsilon=1e-7, **kwargs):
        super(BatchNormalizationLayer, self).__init__(incoming, **kwargs)
        self.epsilon = epsilon

    def get_output_for(self, input, *args, **kwargs):
    	mu = T.mean(input, axis=0)
    	var = T.var(input, axis=0)
    	return (input-mu)/T.sqrt(var+self.epsilon)
