from __future__ import division
from __future__ import print_function
import sys
sys.path.append("/Users/kris/Desktop/ijcai2k18/code/")
from new_code.create_model import *
from new_code.read_data import *
from new_code.train import *
import numpy as np
import argparse
import tensorflow as tf
np.random.seed(1337)
from keras.backend.tensorflow_backend import set_session



config = tf.ConfigProto ()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session (tf.Session (config=config))

def main():

    parser = argparse.ArgumentParser (add_help=True)
    parser.add_argument ("--sampler", choices=['random', 'entropy', 'loss', 'combined', 'ssgd', 'flid'])
    parser.add_argument("--optimizer", choices=['Greedy', 'LazyGreedy', 'ProbGreedy'])
    parser.add_argument ("--num_exp", type=int, default=1)
    parser.add_argument ("--img_folder", type=str)
    parser.add_argument ("--num_epoch", type=int, default=10)
    parser.add_argument ("--steps_per_epoch", type=int, default=None)
    parser.add_argument ("--batch_size", type=int, default=50)
    parser.add_argument("--approx_factor", type=int, default=1)
    parser.add_argument ("--fwd_batch_size", type=int, default=1024)
    parser.add_argument ("--loss_function", type=str, default="categorical_crossentropy")
    parser.add_argument ("--dataset", type=str, choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'svnh', 'ptb'])
    parser.add_argument ("--kernel", type=str, choices=["l2", "fro", "cosine"])
    parser.add_argument ("--folder", type=str, default="./mnist/random/")
    parser.add_argument ("--verbose", type=bool, default=True)

    args = parser.parse_args ()
    x_train, y_train, x_test, y_test = read_data (args.dataset)
    if args.steps_per_epoch == None:
        args.steps_per_epoch = (x_train.shape[0] // args.batch_size)
    model = create_model (x_train.shape[1:], y_train.shape[1], args.loss_function, args.dataset)
    train_model (model, x_train, y_train, x_test, y_test,
                 args.dataset, args.batch_size, args.approx_factor, args.fwd_batch_size,
                 args.loss_function, args.num_epoch,
                 args.num_exp, args.sampler, args.optimizer, args.steps_per_epoch, args.kernel, args.folder)

if __name__ == "__main__":
    main()