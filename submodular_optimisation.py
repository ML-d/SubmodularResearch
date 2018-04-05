import numpy as np
import operator
import sys
import bisect
import heapq
from collections import  OrderedDict

from operator import itemgetter
from abc import ABC, abstractmethod

class Optimisation (ABC):
    def __init__(self, X, Y, fwd_batch_size, batch_size, approx_factor):
        """
        -------------------------------------------------
        :param X: Set of Data points
        :param Y: Set of Data Labels
        :param fnc: The submodular marginal gain function
        :param fwd_batch: The set of sampled points
        -------------------------------------------------
        """
        self.X = X
        self.Y = Y
        self.cardinality = batch_size
        self.fwd_batch_size = fwd_batch_size
        self.sample_points = []
        self.candidate_points = []
        self.k = approx_factor

    def softmax(self, x, t=0.01):
        """
            Compute softmax values for each sets of scores in x.
            Variables
            --------------------------------------------------
            :param x: Variable to compute the softmax value for.
            :
            ---------------------------------------------------
            Probabilities for classes.
        """

        x = list (map (lambda i: np.exp ((1.0 / float (t)) * i), x))
        return x / np.sum (x)

    def create_fwd_batch(self):
        """
        Computes the fwd_batch
        :return: Subsampled data points to make selection from.
        """
        t = np.setdiff1d(np.arange (0, self.X.shape[0]), self.sample_points)
        self.candidate_points = np.random.choice (t, size= self.fwd_batch_size, replace=False)
        assert (len(self.candidate_points)==self.fwd_batch_size)

    @abstractmethod
    def sample(self, candidate_points):
        """
            Sample Elements according to defined strategy
            :return:
        """
        raise Exception ("Not Implemented Error")


class Greedy (Optimisation):
    """
        This implementation is equivalent to the lazier than lazy greedy algorithm.
    """

    def __int__(self, X, Y, fwd_batch_size, batch_size, approx_factor):
        """
            -------------------------------------------------
            :param X: Set of Data points
            :param Y: Set of Data Labels
            :param fnc: The submodular marginal gain function
            :param fwd_batch: The set of sampled points
            -------------------------------------------------
        """
        super (Greedy, self).__init (X, Y, fwd_batch_size, batch_size, approx_factor)

    def sample(self, model, fnc):
        """

        :param max_val:
        :return:
        """
        temp = []
        val = []
        for i in range(0, self.k):
            self.create_fwd_batch ()
            compute_entropy = 1
            for j in self.candidate_points:
                val.append((j, fnc (j, model, self. candidate_points, self.sample_points, compute_entropy)))
                compute_entropy = 0
            sorted(val, key=itemgetter(1), reverse=True)
            keys = [i[0] for i in val]
            self.sample_points.extend (keys[0:self.cardinality// self.k])
            temp.extend(keys[0:self.cardinality// self.k])
        return temp

class dictionary_heap(object):

    def __init__(self):
        self.d = []

    def __getitem__(self, item):
        return dict(self.d)[item]

    def __setitem__(self, key, val):
        # It insert item in an sorted way.
        temp = dict(self.d)
        if key in temp:
            self.insert(key, val)
        else:
            raise("Key Error. Not found {s}".format(s=key))

    def __contains__(self, item):
        temp = dict(self.d)
        if item in temp:
            return  True
        else:
            return  False

    def __iter__(self):
        return iter(self.d)

    def max(self):
        print(self.d)
        return max(self.d, key=itemgetter(1))[1]

    def update(self, item):
        if isinstance(item, tuple):
            self.d.append(item)
        else:
            raise ("Item should be a tuple. Item is of type{s}".format(s=type(item)))

    def pop(self):
        temp = self.d[0]
        print("temp", temp)
        self.d.pop(0)
        return  temp

    def sort(self, reverse=True):
        print(self.d)
        self.d = sorted(self.d, key=itemgetter(1), reverse=reverse)

    def insert(self, key, val):
        keys = [a[0] for a in self.d]
        idx = keys.index(key)
        self.d.pop(idx)
        keys = [a[0] for a in self.d]
        idx = bisect.bisect(keys, key)
        self.d.insert(idx, (key, val))


# Todo fix this
class LazyGreedy (Optimisation):
    """
        This implementation is equivalent to lazier than lazy greedy.
    """

    def __init__(self, X, Y, fwd_batch_size, batch_size):
        """
        -------------------------------------------------
        :param X: Set of Data points
        :param Y: Set of Data Labels
        :param fnc: The submodular marginal gain function
        :param fwd_batch: The set of sampled points
        -------------------------------------------------
        """
        self.priority_queue = {}
        super (LazyGreedy, self).__init__ (X, Y, fwd_batch_size, batch_size)

    def create_priority_queue(self):
        self.priority_queue = dict (zip (np.arange (0, self.X.shape[0]), [sys.maxsize] * self.X.shape[0]))

    def sample(self, model, fnc):
        """
        -----------------------------------------------------------------------------------

        Create a heap of all the elements in the subsampled fwd_batch.
        Heap consists of the following element (alpha, index, freshness)
        alpha: Marginal Gain for elements with index = index
        index: index of the element in X
        freshness: 0/1 indicates if the elements if fresh or not.

        Link: http://j.ee.washington.edu/~bilmes/classes/ee596b_spring_2014/lecture19.pdf
        Using Lazy Greedy algorithm along lazier than lazy greedy.
        Compute the set of points S such that argmax_S:|S|<k f(S)
        Algo
        ---
        1. Pop data point from heap
        2. Check if the element for freshness
        3. if not fresh update the alpha value
        4. if fresh or alpha value > max(heap) then add to sampled points
        5. Else add to the the heap again.
        ------------------------------------------------------------------------------------
        :return: set of k data points that maximisize the given function S
        """
        if (len(self.priority_queue) == 0):
            self.create_priority_queue()

        temp = []
        while(len(temp)<self.cardinality):
            self.create_fwd_batch ()
            mini_dict = dictionary_heap()
            for i in self.candidate_points:
                mini_dict.update((i, self.priority_queue[i]))

            mini_dict.sort(reverse=True)
            while(1 == 1):
                # Sample the highest key
                k = mini_dict.pop()[0]
                # Freshen it up.
                alpha = fnc(k, model, self.candidate_points, self.sample_points)
                # If alpha is greater than everbody else in the sub-sampled set sample it and break.
                if alpha > mini_dict.max():
                    self.sample_points.append (k)
                    temp.append(k)
                    break
                else:
                    mini_dict.update((k, alpha))
            # Update the priority queue
            for k,v in mini_dict:
                self.priority_queue[k] = v
        return temp


class ProbGreedy (Optimisation):

    def __init__(self, X, Y, fwd_batch_size, batch_size):
        """
        -------------------------------------------------
        :param X: Set of Data points
        :param Y: Set of Data Labels
        :param fnc: The submodular marginal gain function
        :param fwd_batch: The set of sampled points
        -------------------------------------------------
        """

        super (ProbGreedy, self).__init__  (X, Y, fwd_batch_size, batch_size)

    def sample(self, model, fnc):
        """
        Using the PD-Greedy Algorithm for selection of
        items based on probablistic greedy algorithm.
        Algo
        ----
        1. For all the points not yet sampled V/X
        2. calculate the marginal fain for each point.
        3. Convert the marginal gains to probability using softmax function.
        4. sample a data point according the probability distribution.
        :return: set of k data points that maximisize the given function S
        """
        temp = []

        for i in range (0, self.k):
            self.create_fwd_batch()
            prob_candidate_points = np.zeros(self.candidate_points.shape)
            for k, v in enumerate (self.candidate_points):
                prob_candidate_points[k] = fnc (v, model, self.candidate_points, self.sample_points)

            prob_candidate_points = self.softmax (prob_candidate_points)
            t = np.random.choice (self.candidate_points, size=self.cardinality // self.k, p=prob_candidate_points)
            temp.extend(t)
            self.sample_points.extend(t)
        return temp