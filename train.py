from new_code.policies import *

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
                dataset, batch_size, fwd_batch_size, loss_function,
                num_epoch, num_exp, sampler, optimizer,
                steps_per_epoch, folder):
    """

    :param model: Neural Network Model
    :param x_train: Array of training data points
    :param y_train: Array of training data labels
    :param x_test: Array of test data points
    :param y_test: Array of test labels
    :param dataset: Name of the dataset
    :param batch_size: Batch size for training
    :param loss_function: loss functions for nn
    :param num_epoch:  Number of passes throught he whole data set.
    :param num_exp: Number times experiments is repeated
    :param sampler: Sampler type
    :param steps_per_epoch: Number of times to do one pass with batch
    :param folder: folder to store the results in.
    :return:
    """
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

        if optimizer == "Greedy":
            optimizer = Greedy (x_train, y_train, fwd_batch_size, batch_size)
        elif optimizer == "LazyGreedy":
            optimizer = LazyGreedy (x_train, y_train, fwd_batch_size, batch_size)
        elif optimizer == "ProbGreedy":
            optimizer = ProbGreedy (x_train, y_train, fwd_batch_size, batch_size)

        if sampler == 'ssgd':
            sampler = SelectSSGD ( x_train, y_train, fwd_batch_size, batch_size, optimizer, loss_function)
        elif sampler == 'random':
            sampler = SelectRandom ( x_train, y_train, fwd_batch_size, batch_size, optimizer, loss_function)
        elif sampler == 'loss':
            sampler = SelectLoss ( x_train, y_train, fwd_batch_size, batch_size, optimizer, loss_function)
        elif sampler == 'entropy':
            sampler = SelectEntropy ( x_train, y_train, fwd_batch_size, batch_size, optimizer, loss_function)
        elif sampler == 'flid':
            sampler = SelectFlid ( x_train, y_train, fwd_batch_size, batch_size, optimizer, loss_function)

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
