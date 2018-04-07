from __future__ import division
from __future__ import print_function
import numpy as np
import six
import argparse
import keras
import gzip
import tensorflow as tf
import pickle
import pandas as pd
import operator

np.random.seed (1337)

from scipy.stats import entropy
import scipy.io as sio

from keras import backend as K
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from keras.models import Model
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, LSTM
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.losses import sparse_categorical_crossentropy
from keras.backend.tensorflow_backend import set_session
from keras.layers import \
    Activation, \
    BatchNormalization, \
    Convolution2D, \
    Dense, \
    Dropout, \
    ELU, \
    Embedding, \
    Flatten, \
    GlobalAveragePooling2D, \
    Input, \
    LSTM, \
    MaxPooling2D, \
    add

from keras.optimizers import SGD

config = tf.ConfigProto ()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session (tf.Session (config=config))


def softmax(x):
    """
        Compute softmax values for each sets of scores in x.
        Variables
        --------------------------------------------------
        x: Variable to compute the softmax value for.
        Return
        ---------------------------------------------------
        Probabilities for classes.
    """
    return np.exp (x) / np.sum (np.exp (x), axis=0)


class SelectLoss:
    """
        Selection based on Loss values of samples.
        No need of rejection sampling.
    """

    def __init__(self, loss):
        """
        :param loss: loss function
        :param x_train: training dataN
        :param y_train: training labels
        """
        self.loss = loss

    def sample(self, model):
        """
            Sort the loss values of the training samples.
            Variables
        """
        idx = np.random.choice (np.arange (0, x_train.shape[0]), size=args.fwd_batch_size, replace=False)
        if args.dataset == "ptb":
            res = model.predict_proba (x_train[idx])
            print ("res.shape", res.shape)
            res = K.get_value (sparse_categorical_crossentropy (tf.convert_to_tensor (y_train[idx], np.float32),
                                                                tf.convert_to_tensor (res)))
            print ("res.shape", res.shape)

        else:
            res = model.predict_proba (x_train[idx])
            print (y_train.shape)
            print (res.shape)
            res = K.get_value (tf.nn.softmax_cross_entropy_with_logits (labels=y_train[idx], logits=res))
        res = res / np.sum (res)
        return np.random.choice (idx,
                                 size=args.batch_size,
                                 replace=False,
                                 p=res)


class SelectRandom:
    """
        Class provides selection based on random sampling
    """

    def __init__(self, loss):
        """
        :param loss: loss function
        :param x_train: training data
        :param y_train: training labels
        """
        self.loss = loss

    def sample(self, model):
        """
        Sample randomly
        ----------------------------------------------

        model: Model for getting losses
        x_train: Full Training Samples
        """
        idx = np.random.choice (np.arange (0, x_train.shape[0]), size=args.fwd_batch_size, replace=False)
        idx = np.random.choice (idx, replace=False,
                                size=args.batch_size)
        return idx


def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train the model based on sampling based on loss
    TODO:Change later to use any loss
    Variable
    ---------------------------------------------------
    :param model: Neural Network Model
    :param x_train: Training Data
    :param y_train: Training Label
    :param x_test: Test Data
    :param y_test: Test Label
    """
    num_exp = args.num_exp
    for exp_num in range (0, num_exp):
        num_epoch = args.num_epoch
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        # burn in epoch 10% of total number of epoch's
        burn_in_epoch = num_epoch // 10
        temp_idx = np.random.choice (np.arange (0, x_train.shape[0]), size=args.batch_size, replace=False)

        model.fit (x_train[temp_idx], y_train[temp_idx], batch_size=args.batch_size, epochs=burn_in_epoch)

        if args.sampler == 'entropy':
            sampler = SelectEntropy (args.loss_function)
        elif args.sampler == 'random':
            sampler = SelectRandom (args.loss_function)
        elif args.sampler == 'loss':
            sampler = SelectLoss (args.loss_function)
        elif args.sampler == 'combined':
            sampler = SelectEntropyDistance (args.loss_function)
        # Make selection
        epoch = 0
        num_epoch = args.num_epoch
        if (args.steps_per_epoch == None):
            steps_per_epoch = (x_train.shape[0] // args.batch_size)
            print ("step_per_epoch", steps_per_epoch)
        else:
            steps_per_epoch = args.steps_per_epoch
        while epoch < num_epoch:
            # Importance sampling is done here
            for ab in range (steps_per_epoch):

                idxs = sampler.sample (model)
                # Train on the sampled data
                t_loss, t_acc = model.train_on_batch (x_train[idxs], y_train[idxs])
                if (ab % 15 == 0):
                    train_loss.append (t_loss)
                    train_acc.append (t_acc)
                    v_loss, v_acc = model.evaluate (x_test, y_test, batch_size=args.batch_size, verbose=False)
                    val_loss.append (v_loss)
                    val_acc.append (v_acc)
                print (epoch, ab)
            epoch += 1
        print (exp_num, epoch)

        # saving models
        print ("Saving Models")
        print (args.folder + "train_acc_model_")
        train_loss = np.array (train_loss)
        val_loss = np.array (val_loss)
        train_acc = np.array (train_acc)
        val_acc = np.array (val_acc)
        np.save (args.folder + "train_acc_model_" + str (exp_num), train_acc)
        np.save (args.folder + "val_acc_model_" + str (exp_num), val_acc)
        np.save (args.folder + "train_loss_model_" + str (exp_num), train_loss)
        np.save (args.folder + "val_loss_model_" + str (exp_num), val_loss)
        model.save_weights (args.folder + "model_" + str (exp_num) + ".h5")


def read_data(dataset):
    """
    Download the data

    """
    if isinstance (dataset, six.string_types):
        pass
    else:
        raise Exception ("Error")
    if (dataset == "mnist"):
        (x_train, y_train), (x_test, y_test) = mnist.load_data ()
    elif (dataset == "cifar10"):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data ()
    elif (dataset == "cifar100"):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data ()
    elif (dataset == "svnh"):
        data = sio.loadmat ('./svnh/train_32x32.mat')
        x_train, y_train = np.transpose (data['X'], (3, 0, 1, 2)), data['y']
        y_train[np.where (y_train == 10)] = 1
        data = sio.loadmat ('./svnh/test_32x32.mat')
        x_test, y_test = np.transpose (data['X'], (3, 0, 1, 2)), data['y']
        y_test[np.where (y_test == 10)] = 1
    elif (dataset == 'fmnist'):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data ()

    if (dataset == 'ptb'):
        with gzip.open ("/Users/kris/.keras/datasets/ptb/ptb_pickle.gz") as f:
            data = pickle.load (f)
            x_train, y_train = data["train"]
            x_test, y_test = data["test"]
            V = data["vocab"]

    print ('x_train shape:', x_train.shape)
    print (x_train.shape[0], 'train samples')
    print (x_test.shape[0], 'test samples')
    print ('y_train shape:', y_train.shape)

    if dataset != "ptb":
        # Convert class vectors to binary class matrices.
        if len (x_train.shape) < 4:
            x_train = np.expand_dims (x_train, axis=-1)
            x_test = np.expand_dims (x_test, axis=-1)
        assert x_train.shape[1:] == x_test.shape[1:]
        assert len (x_train.shape) == 4
        y_train = keras.utils.to_categorical (y_train, num_classes)
        y_test = keras.utils.to_categorical (y_test, num_classes)
        x_train = x_train.astype ('float32')
        x_test = x_test.astype ('float32')
        x_train /= 255
        x_test /= 255

    return x_train, y_train, x_test, y_test


def create_model(input_shape, output_size):
    print ("Dataset", args.dataset)
    print (" output size", output_size)
    if args.dataset == 'ptb':
        vocab_size = 10000
        output_size = vocab_size
        print ("input_shape", input_shape)
        model = Sequential ([
            Embedding (vocab_size + 1, 64, mask_zero=True,
                       input_length=input_shape[0], name='emb1'),
            LSTM (256, unroll=False, return_sequences=True, name='lstm1'),
            Dropout (0.5),
            LSTM (256, unroll=False, name='lstm2'),
            Dropout (0.5),
            Dense (output_size, name='dense1'),
            Activation ("softmax")
        ])

        model.compile (
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model


parser = argparse.ArgumentParser (add_help=True)
parser.add_argument ("--sampler", choices=['random', 'entropy', 'loss', 'combined'])
parser.add_argument ("--num_exp", type=int, default=1)
parser.add_argument ("--img_folder", type=str)
parser.add_argument ("--num_epoch", type=int, default=10)
parser.add_argument ("--steps_per_epoch", type=int, default=None)
parser.add_argument ("--batch_size", type=int, default=50)
parser.add_argument ("--fwd_batch_size", type=int, default=1024)
parser.add_argument ("--loss_function", type=str, default="categorical_crossentropy")
parser.add_argument ("--dataset", type=str, choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'svnh', 'ptb'])
parser.add_argument ("--kernel", type=str, choices=["l2", "fro"])
parser.add_argument ("--folder", type=str, default="./mnist/random/")
parser.add_argument ("--verbose", type=bool, default=True)

args = parser.parse_args ()
num_classes = 10
x_train, y_train, x_test, y_test = read_data (args.dataset)
num_batches = (x_train.shape[0] / args.batch_size)
model = create_model (x_train.shape[1:], y_train.shape[1])
train_model (model, x_train, y_train, x_test, y_test)

