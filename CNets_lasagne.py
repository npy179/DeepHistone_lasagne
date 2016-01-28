import os
import sys
import timeit

import numpy
import numpy as np
import theano
import theano.tensor as T
import lasagne

import cPickle


def load_data(sequences, labels):
    print "loading data.............."
    sequence = np.load(sequences)#make sure the data type is float 32
    label = np.load(labels)

    num_train = 5000
    num_valid = 2500
    num_test = 2500

    train_sequence = sequence[0:num_train,]
    train_label = label[0:num_train]
    train_set = (train_sequence, train_label)

    valid_sequence = sequence[num_train+1:num_train+num_valid,]
    valid_label = label[num_train+1:num_train+num_valid]
    valid_set = (valid_sequence, valid_label)

    test_sequence = sequence[num_train+num_valid+1:]
    test_label = label[num_train+num_valid+1:]
    test_set = (test_sequence, test_label)

    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        shared_x = T._shared(numpy.asarray(data_x, dtype=theano.config.floatX),borrow=True)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),borrow=True)

        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    return rval

def build_CNets(input_var=None, batch_size, nkerns):
    #input layer
    network = lasagne.layers.InputLayer(shape=(batch_size, 1, 4, 600),
                                        input_var = input_var)
    #Convolution layer 1 with nkerns[0]=320, filter_size=(4, 19)
    network = lasagne.layers.Conv2DLayer(network, num_filters=nkerns[0], filter_size=(4, 19),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform)
    #Maxpooling pool_size = (1,3)
    network = lasagne.layers.MaxPool1DLayer(network,pool_size=(1,3))
    #Dropout with rate 0.5
    network = lasagne.layers.dropout(network,p=0.5)

    #Convolution layer 2 with nkerns[1]=480, filter_size=(1, 11)
    network = lasagne.layers.Conv1DLayer(network, num_filters=nkerns[1],filter_size=(1,11),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform)
    #Maxpooling pool_size = (1, 4)
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(1,4))
    #Dropout with rate 0.5
    network = lasagne.layers.dropout(network, p=0.5)

    #Convolution layer 3 with nkerns[2]=960, filter_size(1, 7)
    network = lasagne.layers.Conv1DLayer(network, num_filters=nkerns[2], filter_size=(1, 7),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform)

    #Maxpooling pool_size = (1, 4), pool_size = (1,4)
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(1,4))
    #Dropout with rate 0.5
    network = lasagne.layers.dropout(network, p=0.5)

    #A fully connected layer 1000 units with 50% dropout as its input
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=1000,
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network,p=0.5),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network

def iterate_minibatches():
    pass


def main():
    pass

if __name__ == '__main__':
    main()
