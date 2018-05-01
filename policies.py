import numpy as np
import sys
import time
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from new_code.submodular_optimisation import *
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from sklearn.preprocessing import  normalize
from keras.losses import sparse_categorical_crossentropy

import inspect
def pv(name):
    record=inspect.getouterframes(inspect.currentframe())[1]
    frame=record[0]
    val=eval(name,frame.f_globals,frame.f_locals)
    print('{0}: {1}'.format(name, val))

def read_distances(dataset, kernel):
    if kernel == "l2":
        dist_matrix = np.load("./distance/" + str(dataset) + "_l2.npy")
    elif kernel == "cosine":
        dist_matrix = np.load("./distance/" + str(dataset) + "_cosine.npy")

    return dist_matrix


class SelectSSGD:
    """
    ---------------------------------------------------------------------------------
    SSGD is a submodular function defined as follows
    SSGD(S) = \sum_{i=0}^|S| E(s_i) + P(s_i) + D(s_i)
    E(s_i): Entropy of the item s. - P(y|w, s_i) log P(y|w, s_i)
    P(s_i): Minimum kernalised distance of the point s_i from the S min \phi(s_i, S)
    D(s_i): Average kernelised distance \frac{1}{|S|} \sum_{j=1}^|S| \phi(s_i, s_j)
    The marginal gain for SSGD is just SSGD(s_i)
    ----------------------------------------------------------------------------------
    link : http://ieeexplore.ieee.org/document/6912976/
    """

    def __init__(self, X, Y, fwd_batch_size, batch_size, optimizer, loss, kernel, dataset, compute_once):
        """
        ----------------------------------------------------------------
        fwd_batch: Indicates sampled points from which to select batch
        entropy: Indicates the entropy of all points fwd_batch
        ----------------------------------------------------------------
        :param X: Set of Data points
        :param Y: Set of Data Labels
        :param loss: loss function

        """
        self.X = X
        self.Y = Y
        self.fwd_batch_size = fwd_batch_size
        self.batch_size = batch_size
        self.loss = loss
        self.entropy = np.zeros(shape=(X.shape[0], 1))
        self.features = []
        self.optimizer = optimizer
        self.kernel = kernel
        self.max_entropy = 1
        self.sum_distance = 0
        self.min_distance = 0
        self.feat_dist = None
        self.fro_min = np.zeros((self.X.shape[0], self.X.shape[0]))
        self.fro_mean = np.zeros((self.X.shape[0], self.X.shape[0]))
        self.dist_matrix = None
        self.dataset = dataset
        self.compute_once = compute_once
        self.compute_once_distance()

    def compute_once_distance(self):
        self.dist_matrix = read_distances(self.dataset, self.kernel)


    def compute_entropy(self, model, candidate_points, sampled_points):
        """
        -------------------------------------------------------------------------------
        Computes the entropy for all of the data points in fwd_batch.
        This needs to be done only once since entropy is independent of sampled points.
        -------------------------------------------------------------------------------
        :return: Dictionary of entropy for all of the data points in fwd_batch.
        """
        start_time = time.time()
        intermediate_layer_model = Model (inputs=model.input,
                                          outputs=[model.get_layer ("prob").output,
                                                   model.get_layer("features").output])

        idx = np.hstack((candidate_points, sampled_points))
        idx = idx.astype("int")
        prob, feat= intermediate_layer_model.predict (self.X[idx])
        ent = np.array([entropy(i) for i in prob]).reshape((len(idx), 1))
        if any(np.isnan(ent)) or any(np.isinf(ent)):
            raise("Error")
        self.entropy[idx] = ent
        self.features = np.empty(shape=((self.X.shape[0], feat.shape[1])))
        self.features[idx] = feat

        end_time = time.time()
        self.entropy = normalize(self.entropy, axis=0)
        tot = int(end_time - start_time)
        print("Forward Pass {a} min {b} sec".format(a=tot // 60, b=tot%60))

    def compute_distance(self, model, candidate_points, sampled_points):
        self.sum_distance = np.zeros ((self.X.shape[0], 1))
        self.min_distance = np.zeros_like (self.sum_distance)
        self.features = normalize(self.features, axis=1)
        feat_candidates = self.features[candidate_points]
        feat_sample = self.features[sampled_points]
        if self.kernel=="cosine":
            start_time = time.time ()
            feat_mat = np.dot(feat_candidates, np.transpose(feat_sample)) #cosine product
            feat_mat = np.ones_like(feat_mat) - feat_mat #cosine distance
            self.sum_distance[candidate_points] = np.expand_dims(feat_mat.sum(axis=1) / feat_mat.shape[1], axis=1)
            self.min_distance[candidate_points] = np.expand_dims(np.min(feat_mat, axis=1), axis=1)
            end_time = time.time()
            tot = int (end_time - start_time)
            print ("Computation time {a} min {b} sec".format (a=tot // 60, b=tot % 60))


    def ent(self, idx, model, candidate_points):
        """
        :return: The entropy of data point index by idx in the original data.
        """
        return self.entropy[idx] / float(self.max_entropy)

    def distance(self, idx, candidate_points, sampled_points):
        """
        -------------------------------------------------------------------------------
        Compute the distance term for no duplicates.
        and Compute the diversity term of el based on the sampled points.
        distance = min \phi(el, sampled_points)
        diversity = \frac{1}{N} \sum \phi(el, sampled_points)
        Algo
        ----
        1. If length of sampled points is 0 then return 0
        2. Else Compute the min distance from the sampled points
        -------------------------------------------------------------------------------
        :param el: Candidate point
        :param sampled_points: Set of points already in the selected set.
        :param kernel: kernel function for computing distance
        :return: minimum distance of the candidate from sampled points
        """
        if self.kernel == "cosine":
            if len(sampled_points) == 0:
                return (0, 0)
            else:
                return (self.sum_distance[idx], self.min_distance[idx])

    def marginal_gain(self, idx, model, candidate_points, sampled_points, compute_entropy):
        """
        -------------------------------------------------------------------
        Computes the SSGD value of the given data point indexed by idx
        -------------------------------------------------------------------
        :param data_points: Set of already sampled data points
        :param idx: data point for which to calculate the value
        :return:
        """
        alpha, beta, gamma = 1, 1, 1
        if compute_entropy == 1 :
            self.compute_entropy(model, candidate_points, sampled_points)
        if compute_entropy == 1 and len(sampled_points) > 0 and (self.kernel == "l2" or self.kernel == "cosine"):
            self.compute_distance(model, candidate_points, sampled_points)

        ent = self.ent(idx, model, candidate_points)
        # this computes both distance and diversity
        dist, diversity = self.distance (idx, candidate_points, sampled_points)
        return alpha * ent + beta * diversity + gamma * dist


    def sample(self, model):
        """
        -------------------------------------------------------------------
        Sample the data point according to the following algorithm.
        Algo
        1. Create the set of sub-sampled data points.
        2. Calculate New Entropy values each time new forward batch created.
        3. Use Optimisation method for sampling the data points.
        -------------------------------------------------------------------
        :return Sampled data points of cardianlity k
        """
        return self.optimizer.sample (model, self.marginal_gain)

class SelectEntropy:
    """
    ----------------------------------------------------------------------------------
    Selection based on Entropy
    * I think this is equivalent to uncertainity sampling *
    We define the entropy of a set as the sumation of entropy of
    all the elements.
    E(S) = \sum_{i=1}^N E(s_i) = \sum_{i=1}^N \sum_{j=1}^M P(s_ij) * log (P(s_ij))
    Marginal Gain becomes
    \Del(s_i|S) = E(S\cup s_i) - E(S) = E(s_i)
    This might not be correct since E(S, s_i) = E(s_i) + E(s_i|S)
    ----------------------------------------------------------------------------------
    """

    def __init__(self, X, Y, fwd_batch_size, batch_size, optimizer, loss, _):
        """
        ----------------------------------------------------------------
        fwd_batch: Indicates sampled points from which to select batch
        entropy: Indicates the entropy of all points fwd_batch
        ----------------------------------------------------------------
        :param X: Set of Data points
        :param Y: Set of Data Labels
        :param loss: loss function
        """
        self.loss = loss
        self.fwd_batch_size = fwd_batch_size
        self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.entropy = None
        self.optimizer = optimizer

    def compute_entropy(self, model, candidate_points):
        """
        -------------------------------------------------------------------------------
        Computes the entropy for all of the data points in fwd_batch.
        This needs to be done only once since entropy is independent of sampled points.
        -------------------------------------------------------------------------------
        :return: Dictionary of entropy for all of the data points in fwd_batch.
        """
        self.entropy = dict(zip(candidate_points, map(lambda  x: entropy(x),
                                              model.predict_proba (self.X[candidate_points]))))

    def marginal_gain(self, idx, model, candidate_points, sampled_points, compute_entropy):
        """
        ------------------------------------------------------------
        Computes the entropy of the given data points.
        ------------------------------------------------------------
        :param idx: Element being added
        :return: Entropy of the item intex by idx
        """
        if compute_entropy == 1:
            self.compute_entropy(model, candidate_points)
        return self.entropy[idx]

    def sample(self, model):
        """
        -------------------------------------------------------------------
        Sample the data point according to the following algorithm.
        Algo
        1. Create the set of sub-sampled data points.
        2. Calculate New Entropy values each time new forward batch created.
        3. Use Optimisation method for sampling the data points.
        -------------------------------------------------------------------
        :return Sampled data points of cardianlity k
        """
        return self.optimizer.sample (model, self.marginal_gain)

class SelectFlid:
    """
    --------------------------------------------------------------------------------------------------------
    Facitliy Location submodular functions is defined as follows
    Flid(S) = \sum_{i=1}^|S| u(s_i) + \sum_{d=1}^D (max_{i\in S} (W_{i, d} - \sum{i\in S} W_{i, d})
    u: denotes the modular value. In our case we interpret as negative of probability.
    W: denotes the d-dim reprsentation of the data points. In our case this is equivalent to getting the
        intermidiate reprsentation of the datapoints from the last layer of network.
    --------------------------------------------------------------------------------------------------------
    """

    def __init__(self, X, Y, fwd_batch_size, batch_size, optimizer, loss):
        """
        ----------------------------------------------------------------
        fwd_batch: Indicates sampled points from which to select batch
        entropy: Indicates the entropy of all points in fwd_batch
        features: Indicates the features of all points in fwd_batch
        ----------------------------------------------------------------
        :param X: Set of Data points
        :param Y: Set of Data Labels
        :param loss: loss function
        :param loss:
        """

        self.X = X
        self.Y = Y

        self.loss = loss
        self.candidate_points = []
        self.fwd_batch_size = fwd_batch_size
        self.batch_size =batch_size
        self.entropy = np.array((self.X.shape[0], 1))
        self.features = None
        self.optimizer = optimizer


    def compute_entropy(self, model, candidate_points, sampled_points):
        """
        -------------------------------------------------------------------------------
        Computes the entropy for all of the data points in fwd_batch.
        This needs to be done only once since entropy is independent of sampled points.
        -------------------------------------------------------------------------------
        :return: Dictionary of entropy for all of the data points in fwd_batch.
        """
        start_time = time.time()
        intermediate_layer_model = Model (inputs=model.input,
                                          outputs=[model.get_layer ("prob").output,
                                                   model.get_layer("features").output])

        idx = np.hstack((candidate_points, sampled_points))
        idx = idx.astype("int")
        prob, feat= intermediate_layer_model.predict (self.X[idx])
        ent = np.array([entropy(i) for i in prob]).reshape((len(idx), 1))
        if any(np.isnan(ent)) or any(np.isinf(ent)):
            raise("Error")
        self.entropy[idx] = ent
        self.features = np.empty(shape=((self.X.shape[0], feat.shape[1])))
        self.features[idx] = feat

        end_time = time.time()
        self.entropy = normalize(self.entropy, axis=0)
        tot = int(end_time - start_time)
        print("Forward Pass {a} min {b} sec".format(a=tot // 60, b=tot%60))

    def modular(self, idx):
        """
        :param idx: Element being added
        :return: The entropy of data point index by idx in the original data.
        """
        return self.entropy[idx]

    def distance(self, idx, candidate_points, sampled_points):
        """
        --------------------------------------------------------------
        Compute the diversity according to the following
        D(s) = \sum_{d=1}^D (max_{i\in S} (W_{i, d} - \sum{i\in S} W_{i, d})
        Algo
        1. Compute the D(s).
        2. Compute the Value for D(s+1)
        --------------------------------------------------------------
        :param idx: Element being added
        :param data_points: Already sampled data points
        :return: D(s) - D(s+1)
        """
        if len(sampled_points) == 0:
            return 0
        else:
            feature_matrix = self.features[sampled_points]
            temp = sampled_points.extend(idx)
            feature_matrix_plus_idx = self.features[temp]


        mean = np.sum(feature_matrix, axis=1)
        feature_matrix = feature_matrix - mean
        val_prev = np.sum(np.max(feature_matrix, axis=1))
        mean = np.sum (feature_matrix_plus_idx, axis=1)
        feature_matrix_plus_idx = feature_matrix_plus_idx - mean
        val = np.sum(np.max(feature_matrix_plus_idx, axis=1))
        return val - val_prev

    def marginal_gain(self, idx, model, candidate_points, sampled_points, compute_entropy):
        """
        Computes the entropy of the given data points
        :param idx: Element being added
        :param model: NN model
        :param data_points: already sampled points
        :return:
        """
        if compute_entropy == 1 :
            self.compute_entropy(model, candidate_points, sampled_points)
        return self.modular(idx) + self.distance(idx, candidate_points, sampled_points)

    def sample(self, model):
        """
        Create a forward batch.
        Compute entropy of sampled + forward batch.
        Compute features for sampled + forward batch
        Sample using optimizer.
        Variables
        """
        return  self.optimizer.sample (model, self.marginal_gain, self.candidate_points)

class SelectLoss:
    """
        Selection based on Loss values of samples.
        No need of rejection sampling.
    """

    def __init__(self, X, Y, fwd_batch_size, batch_size, _,  loss, kernel):
        """
        :param loss: loss function
        :param x_train: training dataN
        :param y_train: training labels
        """
        self.X = X
        self.Y = Y
        self.fwd_batch_size = fwd_batch_size
        self.batch_size = batch_size
        self.loss = loss
        self.sample_points = []
        self.A = tf.placeholder('float', shape=[None, 10])
        self.B = tf.placeholder('float', shape=[None, 10])
        self.res = tf.nn.softmax_cross_entropy_with_logits(labels=self.A, logits=self.B)
        self.fnc = K.function([self.A, self.B], [self.res])

    def sample(self, model):
        """
            Sort the loss values of the training samples.
            Variables
        """
        t = np.setdiff1d(np.arange (0, self.X.shape[0]), self.sample_points)
        idx = np.random.choice (t, size=self.fwd_batch_size, replace=False)
        res = model.predict_proba (self.X[idx])
        res = self.fnc([self.Y[idx], res])
        res = res[0] / np.sum(res[0])
        idx = np.random.choice (idx,
                                 size=self.batch_size,
                                 replace=False,
                                 p=res)
        self.sample_points.extend(idx)
        return idx

class SelectRandom:
    """
        Class provides selection based on random sampling
    """

    def __init__(self, X, Y, fwd_batch_size, batch_size, _, loss, kernel):
        """
        :param loss: loss function
        :param x_train: training data
        :param y_train: training labels
        """
        self.X = X
        self.Y = Y
        self.fwd_batch_size = fwd_batch_size
        self.batch_size = batch_size
        self.loss = loss
        self.sample_points = []

    def sample(self, _):
        """
        Sample randomly
        ----------------------------------------------

        model: Model for getting losses
        x_train: Full Training Samples
        """
        t = np.setdiff1d(np.arange (0, self.X.shape[0]), self.sample_points)
        idx = np.random.choice (t, size = self.fwd_batch_size, replace=False)
        idx = np.random.choice (idx, replace=False,
                                size=self.batch_size)
        self.sample_points.extend(idx)
        return idx