from __future__ import division
from __future__ import print_function
from create_model import  *
from read_data import *
from policies import *
from submodular_optimisation import *

import argparse
import tensorflow as tf

np.random.seed(1337)
from keras.backend.tensorflow_backend import set_session



config = tf.ConfigProto ()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session (tf.Session (config=config))




def softmax(x):
    """
        --------------------------------------------------
        Compute softmax values for each sets of scores in x.
        Variables
        ---------------------------------------------------
        :param x: Variable to compute the softmax value for.
        :return: Probabilities for classes.
    """
    x = list (map (lambda i: np.exp (i), x))
    return x / np.sum (x)


def train_model(model, x_train, y_train, x_test, y_test,
                dataset, batch_size, loss_function,
                num_epoch, num_exp, sampler,
                steps_per_epoch, folder):
    """

    :param model:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param dataset:
    :param batch_size:
    :param loss_function:
    :param num_epoch:
    :param num_exp:
    :param sampler:
    :param steps_per_epoch:
    :param folder:
    :return:
    """
    num_exp = num_exp
    for exp_num in range (0, num_exp):
        num_epoch = num_epoch
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        # burn in epoch 10% of total number of epoch's
        burn_in_epoch = num_epoch // 10
        temp_idx = np.random.choice (np.arange (0, x_train.shape[0]), size=batch_size, replace=False)

        model.fit (x_train[temp_idx], y_train[temp_idx], batch_size=batch_size, epochs=burn_in_epoch)

        if sampler == 'ssgd':
            sampler = SelectSSGD (loss_function)
        elif sampler == 'random':
            sampler = SelectRandom (loss_function)
        elif sampler == 'loss':
            sampler = SelectLoss (loss_function)
        elif sampler == 'entropy':
            sampler = SelectEntropy (loss_function)
        elif sampler == 'flid':
            sampler = SelectFlid (loss_function)
        # Make selection
        epoch = 0
        num_epoch = num_epoch
        if (steps_per_epoch == None):
            steps_per_epoch = (x_train.shape[0] // batch_size)
            print ("step_per_epoch", steps_per_epoch)
        else:
            steps_per_epoch = steps_per_epoch
        if dataset == "cifar10":
            while epoch < num_epoch:
                # Importance sampling is done here
                for ab in range (steps_per_epoch):
                    idxs = sampler.sample (model)
                    # Train on the sampled data
                    t_loss, t_acc = model.train_on_batch (x_train[idxs], y_train[idxs])
                    train_loss.append (t_loss)
                    train_acc.append (t_acc)
                print (exp_num, epoch)
                v_loss, v_acc = model.evaluate (x_test, y_test, batch_size=batch_size)
                val_loss.append (v_loss)
                val_acc.append (v_acc)
                print ("Validation Loss", v_loss)
                print ("Validation Acc", v_acc)
                epoch += 1
        else:
            while epoch < num_epoch:
                # Importance sampling is done here
                for ab in range (steps_per_epoch):

                    idxs = sampler.sample (model)
                    print("idxs.shape" , len(idxs))
                    # Train on the sampled data
                    t_loss, t_acc = model.train_on_batch (x_train[idxs], y_train[idxs])
                    if (ab % 15 == 0):
                        train_loss.append (t_loss)
                        train_acc.append (t_acc)
                        v_loss, v_acc = model.evaluate (x_test, y_test, batch_size=batch_size, verbose=False)
                        val_loss.append (v_loss)
                        val_acc.append (v_acc)
                    print (epoch, ab)
                epoch += 1
            print (exp_num, epoch)

        # saving models
        print ("Saving Models")
        print (folder + "train_acc_model_")
        train_loss = np.array (train_loss)
        val_loss = np.array (val_loss)
        train_acc = np.array (train_acc)
        val_acc = np.array (val_acc)
        np.save (folder + "train_acc_model_" + str (exp_num), train_acc)
        np.save (folder + "val_acc_model_" + str (exp_num), val_acc)
        np.save (folder + "train_loss_model_" + str (exp_num), train_loss)
        np.save (folder + "val_loss_model_" + str (exp_num), val_loss)
        model.save_weights (folder + "model_" + str (exp_num) + ".h5")



def main():

    parser = argparse.ArgumentParser (add_help=True)
    parser.add_argument ("--sampler", choices=['random', 'entropy', 'loss', 'combined', 'ssgd', 'flid'])
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
    x_train, y_train, x_test, y_test = read_data (args.dataset)
    num_batches = (x_train.shape[0] / args.batch_size)
    model = create_model (x_train.shape[1:], y_train.shape[1], args.loss_function, args.dataset)
    train_model (model, x_train, y_train, x_test, y_test,
                 args.dataset, args.batch_size,
                 args.loss_function, args.num_epoch,
                 args.num_exp, args.sampler, args.steps_per_epoch, args.folder)
