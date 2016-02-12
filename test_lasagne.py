#!/usr/bin/python
from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
import timeit
import lasagne

def main():


    inputs0 = np.load("seq_test.npy")
    targets = np.load("label_test.npy")

    tic = timeit.default_timer()
    inputs1 = inputs0.reshape(100,1,4,600)

    input_var = T.tensor4("inputs")
    target_var = T.ivector("targets")

    network = lasagne.layers.InputLayer(shape=(100, 1, 4, 600), input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, 320, filter_size=(4, 19),
                                         nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = (1, 3))
    network = lasagne.layers.Conv2DLayer(network, 480, filter_size=(1,11),
                                         nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = (1, 4))
    network = lasagne.layers.Conv2DLayer(network, 960, filter_size = (1, 7),
                                         nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = (1, 4))
    network = lasagne.layers.DenseLayer(network, num_units = 1000, nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(network, num_units = 2, nonlinearity=lasagne.nonlinearities.softmax)

    output = lasagne.layers.get_output(network) # output is 2D list

    l1_penalty = lasagne.regularization.regularize_layer_params(network)

    l1 = lasagne.layers.get_output(l1_penalty)
    print(l1)

    output_shape = lasagne.layers.get_output_shape(network)
    all_parameters = lasagne.layers.get_all_param_values(network)

    test_code = theano.function([input_var], [output])

    output  = test_code(inputs1)

    toc = timeit.default_timer()
    time = toc - tic

    print("time is : ", time)
if __name__=="__main__":
    main()
