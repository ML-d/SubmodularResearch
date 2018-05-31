import six
import keras
import gzip
import scipy.io as sio
import numpy as np
import pickle

from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import fashion_mnist
from keras.datasets import mnist

def create_imbalanced(x_train, y_train, **kwargs):
    """
    The function creates a imbalenced samples.
    This is done using sampling given from class lables
    in a ratio given.
    :param
    kwargs: (class_label, sampling_ratio)
    """

    data = list (zip (np.arange (0, len (y_train)), y_train))
    data = np.array (data)
    idx = []
    for key, value in kwargs.items ():
        key = int (key)
        value = float (value)
        candidates = data[data[:, 1] == key][:, 0]
        idx.extend (np.random.choice (candidates, size=int (len (candidates) * value), replace=False))
    return idx

def read_data(dataset):
    """
    Download the data

    """
    if isinstance (dataset, six.string_types):
        pass
    else:
        raise Exception ("Error")
    if dataset == "cifar100":
        num_classes = 100
    else:
        num_classes = 10

    if (dataset == "mnist"):
        (x_train, y_train), (x_test, y_test) = mnist.load_data ()
    elif (dataset == 'fmnist'):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data ()
    elif (dataset == "cifar10"):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data ()
    elif (dataset == "cifar100"):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data ()
    elif (dataset == "svnh"):
        data = sio.loadmat ('../svnh/train_32x32.mat')
        x_train, y_train = np.transpose (data['X'], (3, 0, 1, 2)), data['y']
        y_train[np.where (y_train == 10)] = 1
        data = sio.loadmat ('../svnh/test_32x32.mat')
        x_test, y_test = np.transpose (data['X'], (3, 0, 1, 2)), data['y']
        y_test[np.where (y_test == 10)] = 1


    if (dataset == 'ptb'):
        with gzip.open ("/Users/kris/.keras/datasets/ptb/ptb_pickle.gz") as f:
            data = pickle.load (f)
            x_train, y_train = data["train"]
            x_test, y_test = data["test"]
            V = data["vocab"]

    if (dataset == "im-mnist"):
        (x_train, y_train), (x_test, y_test) = mnist.load_data ()
        kw = {'0': '0.01', '1': '0.01', '2': '0.01', '3': '0.02', '4': '0.5', '5': '0.011', '6': '0.01', '7': '0.01',
              '8': '0.01', '9': '0.4'}
        idx = np.array (create_imbalanced (x_train, y_train, **kw))
        x_train = x_train[idx]
        y_train = y_train[idx]

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

    print ('x_train shape:', x_train.shape)
    print (x_train.shape[0], 'train samples')
    print (x_test.shape[0], 'test samples')
    print ('y_train shape:', y_train.shape)

    return x_train, y_train, x_test, y_test