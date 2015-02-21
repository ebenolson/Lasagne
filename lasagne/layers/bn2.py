import numpy as np
import theano
import theano.tensor as T
from .. import init
from .. import utils
from .. import nonlinearities

from .base import Layer


__all__ = [
    "BatchNormLayer2"
]


class BatchNormLayer2(Layer):

    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, input_layer,
                 gamma=init.Uniform([0.95, 1.05]),
                 beta=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 epsilon=0.001,
                 **kwargs):
        super(BatchNormLayer2, self).__init__(input_layer, **kwargs)
        self.additional_updates = None
        self.epsilon = epsilon
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        input_shape = input_layer.get_output_shape()

        if len(input_shape) == 2:       # in case of dense layer
            self.axis = (0)
            param_shape = (input_shape[-1])
            self.gamma = self.create_param(gamma, param_shape)
            self.beta = self.create_param(beta, param_shape)
            ema_shape = (1, input_shape[-1])
            ema_bc = (True, False)
        elif len(input_shape) == 4:     # in case of conv2d layer
            self.axis = (0, 2, 3)
            param_shape = (input_shape[1], 1, 1)

            # it has to be made broadcastable on the first axis
            self.gamma = theano.shared(utils.floatX(gamma(param_shape)),
                                       broadcastable=(False, True, True),
                                       borrow=True)
            self.beta = theano.shared(utils.floatX(beta(param_shape)),
                                      broadcastable=(False, True, True),
                                      borrow=True)
            ema_shape = (1, input_shape[1], 1, 1)
            ema_bc = (True, False, True, True)
        else:
            raise NotImplementedError

        self.mean_ema = theano.shared(
            np.zeros(ema_shape, dtype=theano.config.floatX),
            borrow=True, broadcastable=ema_bc)

        self.variance_ema = theano.shared(
            np.ones(ema_shape, dtype=theano.config.floatX),
            borrow=True, broadcastable=ema_bc)

        # TODO: REMOVE
        self.aux = theano.shared(
            np.ones(ema_shape, dtype=theano.config.floatX),
            borrow=True, broadcastable=ema_bc)

        self.batch_cnt = theano.shared(0)

    def get_params(self):
        return [self.gamma, self.beta]

    def get_state(self):
        return [self.gamma, self.beta,
                self.mean_ema, self.variance_ema, self.batch_cnt]

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, deterministic=False, *args, **kwargs):

        if deterministic:
            m = self.mean_ema
            v = self.variance_ema
        else:
            m = T.mean(input, self.axis, keepdims=True)
            v = T.sqrt(T.var(input, self.axis, keepdims=True) + self.epsilon)
            self.additional_updates = [
                # (self.mean_ema, self.mean_ema * 0.98 + m * 0.02),
                # (self.variance_ema, self.variance_ema * 0.98 + m * 0.02)]
                (self.mean_ema, self.mean_ema + m),
                (self.variance_ema, self.variance_ema + v),
                (self.batch_cnt, self.batch_cnt + 1)]

        input_norm = (input - m) / v
        y = self.gamma * input_norm + self.beta

        return self.nonlinearity(y)

    def get_additional_updates(self):
        if not self.additional_updates:
            raise RuntimeError
        return self.additional_updates

    def pre_train(self):
        self.mean_ema.set_value(np.zeros(
            self.mean_ema.get_value().shape,
            dtype=theano.config.floatX))
        self.variance_ema.set_value(np.zeros(
            self.variance_ema.get_value().shape,
            dtype=theano.config.floatX))
        self.batch_cnt.set_value(0)

    def post_train(self):
        new_mean = self.mean_ema.get_value() / self.batch_cnt.get_value()
        self.mean_ema.set_value(new_mean)
        new_var = self.variance_ema.get_value() / self.batch_cnt.get_value()
        self.variance_ema.set_value(new_var)
