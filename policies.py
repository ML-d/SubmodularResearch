import numpy as np
import sys
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from new_code.submodular_optimisation import *
from scipy.stats import entropy
from keras.losses import sparse_categorical_crossentropy


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

    def __init__(self, X, Y, fwd_batch_size, batch_size, loss):
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
        self.candidate_points = []
        self.entropy = None


    def create_batch(self):
        """
        Computes the fwd_batch
        :return: Subsampled data points to make selection from.
        """
        self.candidate_points = np.random.choice (np.arange (0, self.X.shape[0]), size= self.fwd_batch_size)

    def compute_entropy(self, model):
        """
        -------------------------------------------------------------------------------
        Computes the entropy for all of the data points in fwd_batch.
        This needs to be done only once since entropy is independent of sampled points.
        -------------------------------------------------------------------------------
        :return: Dictionary of entropy for all of the data points in fwd_batch.
        """
        self.entropy = dict(zip(self.candidate_points, map(lambda  x: entropy(x),
                                              model.predict_proba (self.X[self.candidate_points]))))

    def ent(self, idx):
        """
        :return: The entropy of data point index by idx in the original data.
        """
        if self.entropy == None:
            self.compute_entropy ()
        return self.entropy[idx]

    def distance(self, idx, sampled_points, kernel):
        """
        -------------------------------------------------------------------------------
        Compute the distance term for no duplicates.
        distance = min \phi(el, sampled_points)
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
        min_dist = 1000000
        if kernel == "l2":
            if len(sampled_points) == 0:
                return 0
            for i in sampled_points:
                min_dist = min (np.linalg.norm (np.squeeze(self.X[idx], 2) - np.squeeze(self.X[i], 2)), min_dist)
                print(min_dist)

        if kernel == "fro":
            if len(sampled_points) == 0:
                return 0
            for i in sampled_points:
                min_dist = min (np.linalg.norm (np.squeeze(self.X[idx], 2) - np.squeeze(self.X[i], 2), "fro"),
                                min_dist)
                print ("min_dist", min_dist)

        return min_dist

    def diversity(self, idx, sampled_points, kernel):
        """
        --------------------------------------------------------------------------------
        Compute the diversity term of el based on the sampled points.
        diversity = \frac{1}{N} \sum \phi(el, sampled_points)
        --------------------------------------------------------------------------------
        :param el: Candidate point
        :param sampled_points: Set of points already in the selected set.
        :return: average kernelized distance of the candidate from sampled points

        """
        if kernel == "l2":
            dist = 0
            if len(sampled_points) == 0:
                return dist
            for i in sampled_points:
                dist += np.linalg.norm (np.squeeze(self.X[idx], 2) - np.squeeze(self.X[i], 2))
            dist / len (sampled_points)

        if kernel == "fro":
            dist = 0
            if len(sampled_points) == 0:
                return dist
            for i in sampled_points:
                dist += np.linalg.norm (np.squeeze(self.X[idx], 2) - np.squeeze(self.X[i], 2), "fro")
            dist / len (sampled_points)
            print ("len(sampled_points)", len(sampled_points))

        return dist

    def marginal_gain(self, idx, data_points):
        """
        -------------------------------------------------------------------
        Computes the SSGD value of the given data point indexed by idx
        -------------------------------------------------------------------
        :param data_points: Set of already sampled data points
        :param idx: data point for which to calculate the value
        :return:
        """
        ent = self.ent(idx)
        diversity = self.diversity (idx, data_points)
        dist = self.distance (idx, data_points)
        print("ent, diversity, dist", ent, diversity, dist)
        return ent + diversity + dist

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
        self.forward_batch_size()
        self.entropy = None
        optimizer = LazyGreedy (self.X, self.Y, self.marginal_gain, self.candidate_points, self.batch_size)
        optimizer.sample ()
        return optimizer.sample_points


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

    def __init__(self, X, Y, fwd_batch_size, batch_size, loss):
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
        self.candidate_points = []
        self.entropy = None

    def create_fwd_batch(self):
        """
        -------------------------------------------------------------------------------
        Computes the entropy for all of the data points in fwd_batch.
        This needs to be done only once since entropy is independent of sampled points.
        -------------------------------------------------------------------------------
        :return: Dictionary of entropy for all of the data points in fwd_batch.
        """
        self.candidate_points = np.random.choice (np.arange (0, self.X.shape[0]), size= self.fwd_batch_size)

    def compute_entropy(self, model):
        """
        -------------------------------------------------------------------------------
        Computes the entropy for all of the data points in fwd_batch.
        This needs to be done only once since entropy is independent of sampled points.
        -------------------------------------------------------------------------------
        :return: Dictionary of entropy for all of the data points in fwd_batch.
        """
        self.entropy = dict(zip(self.candidate_points, map(lambda  x: entropy(x),
                                              model.predict_proba (self.X[self.candidate_points]))))

    def marginal_gain(self, idx, _):
        """
        ------------------------------------------------------------
        Computes the entropy of the given data points.
        ------------------------------------------------------------
        :param idx: Element being added
        :return: Entropy of the item intex by idx
        """
        if self.entropy == None:
            self.compute_entropy()
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
        self.create_fwd_batch()
        self.entropy = None
        optimizer = LazyGreedy (self.X, self.Y, self.marginal_gain, self.candidate_points, self.batch_size)
        optimizer.sample (model)
        return optimizer.sample_points


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

    def __init__(self, X, Y, fwd_batch_size, batch_size, loss):
        """
        ----------------------------------------------------------------
        fwd_batch: Indicates sampled points from which to select batch
        entropy: Indicates the entropy of all points in fwd_batch
        features: Indiactes the features of all points in fwd_batch
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
        self.entropy = None
        self.features = None


    def create_fwd_batch(self):
        """
        -------------------------------------------------------------------------------
        Computes the entropy for all of the data points in fwd_batch.
        This needs to be done only once since entropy is independent of sampled points.
        -------------------------------------------------------------------------------
        :return: Dictionary of entropy for all of the data points in fwd_batch.
        """
        self.candidate_points = np.random.choice (np.arange (0, self.X.shape[0]), size= self.fwd_batch_size)

    def compute_entropy(self, model):
        """
        -------------------------------------------------------------------------------
        Computes the entropy for all of the data points in fwd_batch.
        This needs to be done only once since entropy is independent of sampled points.
        -------------------------------------------------------------------------------
        :return: Dictionary of entropy for all of the data points in fwd_batch.
        """
        self.entropy = dict(zip(self.candidate_points, map(lambda  x: -entropy(x),
                                              model.predict_proba (self.X[self.candidate_points]))))

    def compute_features(self, model):
        """"
        -------------------------------------------------------------------------------
        Computes the features for all of the data points in fwd_batch.
        This needs to be done only once since features is independent of sampled points.
        -------------------------------------------------------------------------------
        :return: Dictionary of entropy for all of the data points in fwd_batch.
        """
        intermediate_layer_model = Model (inputs=model.input,
                                          outputs=model.get_layer("features").output)
        self.features = dict(zip(self.candidate_points, intermediate_layer_model.predict(self.X[self.candidate_points])))

    def modular(self, idx):
        """
        :return: The entropy of data point index by idx in the original data.
        """
        if self.entropy == None:
            self.compute_entropy()

        return self.entropy[idx]


    def distance(self, idx, data_points):
        """
        --------------------------------------------------------------
        Compute the diversity according to the following
        D(s) = \sum_{d=1}^D (max_{i\in S} (W_{i, d} - \sum{i\in S} W_{i, d})
        Algo
        1. Compute the D(s).
        2. Compute the Value for D(s+1)
        --------------------------------------------------------------
        :return: D(s) - D(s+1)
        """
        if len(data_points) == 0:
            return np.sum(self.features[idx])

        feature_matrix = np.empty(self.features[data_points[0]].shape)
        print ("feature_matrix.shape", feature_matrix.shape)

        tot = 0
        for i in data_points:
            feature_matrix = np.hstack([feature_matrix, self.features[i]])

        print(self.features[data_points[0]].shape)

        print(feature_matrix.shape)
        tot = np.sum(feature_matrix, axis=1)
        temp_matrix = feature_matrix - tot
        max_col = np.max(temp_matrix, axis=1)

        feature_matrix = np.hstack([self.features[idx], feature_matrix])

        tot = np.sum (feature_matrix, axis=1)
        temp_matrix = feature_matrix - tot
        new_max_col = np.max(temp_matrix, axis=1)

        return np.sum(new_max_col) - np.sum(max_col)



    def marginal_gain(self, idx, model, data_points):
        """
        Computes the entropy of the given data points
        :param idx: Element being added
        :return:
        """
        if self.entropy == None:
            self.compute_entropy(model)
        if self.features == None:
            self.compute_features(model)
        return self.modular(idx) + self.distance(idx, data_points)

    def sample(self, model):
        """
        Sort the loss values of the training samples.
        Variables
        """
        self.create_fwd_batch()
        self.entropy = None
        self.features = None
        optimizer = LazyGreedy (self.X, self.Y, self.marginal_gain, self.candidate_points, self.batch_size)
        optimizer.sample (model)
        return optimizer.sample_points



class SelectLoss:
    """
        Selection based on Loss values of samples.
        No need of rejection sampling.
    """

    def __init__(self, X, Y, fwd_batch_size, batch_size, loss):
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

    def sample(self, model, batch_size, dataset):
        """
            Sort the loss values of the training samples.
            Variables
        """
        idx = np.random.choice (np.arange (0, self.X.shape[0]), size=self.fwd_batch_size, replace=False)
        if dataset == "ptb":
            res = model.predict_proba (self.X[idx])
            res = K.get_value (sparse_categorical_crossentropy (tf.convert_to_tensor (self.Y[idx], np.float32),
                                                                tf.convert_to_tensor (res)))
            print ("res.shape", res.shape)

        else:
            res = model.predict_proba (self.X[idx])
            print (self.Y.shape)
            print (res.shape)
            res = K.get_value (tf.nn.softmax_cross_entropy_with_logits (labels=self.Y[idx], logits=res))
        res = res / np.sum (res)

        return np.random.choice (idx,
                                 size=batch_size,
                                 replace=False,
                                 p=res)


class SelectRandom:
    """
        Class provides selection based on random sampling
    """

    def __init__(self, X, Y, fwd_batch_size, loss):
        """
        :param loss: loss function
        :param x_train: training data
        :param y_train: training labels
        """
        self.X = X
        self.Y = Y
        self.fwd_batch_size = fwd_batch_size
        self.loss = loss

    def sample(self, batch_size):
        """
        Sample randomly
        ----------------------------------------------

        model: Model for getting losses
        x_train: Full Training Samples
        """
        idx = np.random.choice (np.arange (0, self.X.shape[0]), size = self.fwd_batch_size, replace=False)
        idx = np.random.choice (idx, replace=False,
                                size=batch_size)
        return idx