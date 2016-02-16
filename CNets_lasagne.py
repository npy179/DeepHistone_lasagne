from __future__ import print_function
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
    print("loading data .........")
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
        shared_y = T._shared(numpy.asarray(data_y, dtype=theano.config.floatX),borrow=True)

        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    return rval

    #return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y

def build_CNets(batch_size, nkerns, input_var):
    #input layer
    network = lasagne.layers.InputLayer(shape=(batch_size, 1, 4, 600),
                                        input_var = input_var)
    #Convolution layer 1 with nkerns[0]=320, filter_size=(4, 19)
    network = lasagne.layers.Conv2DLayer(network, num_filters=nkerns[0], filter_size=(4, 19),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform())
    #Maxpooling pool_size = (1,3)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 3))
    #Dropout with rate 0.5
    network = lasagne.layers.dropout(network,p = 0.5)

    #Convolution layer 2 with nkerns[1]=480, filter_size=(1, 11)
    network = lasagne.layers.Conv2DLayer(network, num_filters=nkerns[1], filter_size=(1,11),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform())
    #Maxpooling pool_size = (1, 4)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1,4))
    #Dropout with rate 0.5
    network = lasagne.layers.dropout(network, p=0.5)

    #Convolution layer 3 with nkerns[2]=960, filter_size(1, 7)
    network = lasagne.layers.Conv2DLayer(network, num_filters=nkerns[2], filter_size=(1, 7),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform())

    #Maxpooling pool_size = (1, 4), pool_size = (1,4)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 4))
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

####################################################
#             Batch iterator                       #
# input is training dataset (sequence, target )    #
####################################################

def main(batch_size = 100, nkerns = [320, 480, 960]):
    #load data
    print("loading data...................")
    sequences = "H3K27me3_sequence_fore_back_10000.npy"
    labels = "H3K27me3_label_fore_back_10000.npy"

    datasets = load_data(sequences=sequences, labels=labels)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute # of minibatches for training, validation and testing
    n_train_bathes = train_set_x.get_value(borrow=True).shape[0]
    n_valid_bathes = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    n_train_bathes /= batch_size
    n_valid_bathes /= batch_size
    n_test_batches /= batch_size

    # index to a batch
    index = T.lscalar()
    #Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    input_var_0layer = input_var.reshape((batch_size, 1, 4, 600))
    target_var = T.ivector('targets')

    #Create neural network model
    #input_var0 = input_var.reshape(batch_size, 1, 4, 600)## here I have some problem
    network = build_CNets(batch_size, nkerns, input_var_0layer)

    #############################################################
    #Create objective functions
    #Create loss expression for training
    #############################################################
    prediction = lasagne.layers.get_output(network) # Check the prediction and target value
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    #Create regularization term L1, L2
    lambda1 = 5e-07
    lambda2 = 1e-08
    l1_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l1)*lambda1 #how to use regularize_network_params
    l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)*lambda2

    loss = loss + l1_penalty + l2_penalty

    #Create update expression for training SGD with Nesterov momentum
    params = lasagne.layers.get_all_params(network,trainable=True)
    updates = lasagne.updates.adagrad(loss, params, learning_rate=0.03,epsilon=1e-06)

    #Create loss function for validatation/testing, we do a deterministic forward pass through the network, disabling dropout layers
    test_prediction = lasagne.layers.get_output(network, deterministic = True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    #Create classfication accuracy
    test_acc = T.mean(T.eq(T.argmax(test_prediction,axis=1), target_var),dtype=theano.config.floatX)

    #Compile a function performing training step on minibatch of training dataset

    print("compiling training function ..........")
    train_model = theano.function(
        [index],
        loss,
        updates=updates,
        givens={
            input_var: train_set_x[index * batch_size: (index + 1) * batch_size],
            target_var: train_set_y[index * batch_size: (index + 1) * batch_size]
        })


    #Compile a function computing accuracy of validation data
    print("compiling validation function ............")
    validate_model = theano.function(
        [index],
        [test_acc],
        givens={
            input_var: valid_set_x[index * batch_size: (index + 1) * batch_size],
            target_var: valid_set_y[index * batch_size: (index + 1) * batch_size]
        })

    # Compile a function performing test step on minibatch of test data
    print("compiling test function .............")
    test_model = theano.function(
        [index],
        [test_acc],
        givens={
            input_var: test_set_x[index * batch_size: (index + 1) * batch_size],
            target_var: test_set_y[index * batch_size: (index +1) * batch_size]
        })



    #Start training
    print("Starting training...................")
    # early-stopping parameters
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    n_train_batches = 200 # there are 200 batchs in training dataset
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss =numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    n_epochs = 1000

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):# minibatch_index 0,1,2,3,....,199
            iter = (epoch - 1) * n_train_batches + minibatch_index # epoch is 1: iter each minibatch is one iter
            #if iter % 100 == 0:
            #    print("training @ iter", iter) #100, 200, 300, 400
            cost_ij = train_model(minibatch_index) # input the mini_batch index, after iterate all the case in this mini batch, it will give out one cost_ij

            if (iter+1) % validation_frequency == 0: # iter = 199
                #compute loss on validation dataset (all batches)
                validation_loss = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_loss)

                if this_validation_loss < best_validation_loss:

                    #save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    #test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_bathes)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    with open("best_model.pkl","w") as f:
                        cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
            if patience <= iter:
                done_looping = True
                break


    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    main()
