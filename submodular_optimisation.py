import numpy as np
import operator
import heapq

from operator import itemgetter
from abc import ABC, abstractmethod

class Optimisation (ABC):
    def __init__(self, X, Y, fwd_batch_size, batch_size):
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

    def __int__(self, X, Y, fnc, batch_size):
        """
            -------------------------------------------------
            :param X: Set of Data points
            :param Y: Set of Data Labels
            :param fnc: The submodular marginal gain function
            :param fwd_batch: The set of sampled points
            -------------------------------------------------
        """
        super (Greedy, self).__init (X, Y, batch_size)

    def sample(self, model, fnc):
        """

        :param max_val:
        :return:
        """
        temp = []
        for i in range (0, self.cardinality):
            max_val = -1000000
            self.create_fwd_batch ()
            for j in self.candidate_points:
                present_val = fnc (j, model, self. candidate_points, self.sample_points)
                if present_val > max_val:
                    max_idx = j
                    max_val = present_val
            self.sample_points.append (max_idx)
            temp.append(max_idx)
        return temp

# Todo fix this
class LazyGreedy (Optimisation):
    """
        This implementation is equivalent to lazier than lazy greedy.
    """

    def __init__(self, X, Y, batch_size):
        """
        -------------------------------------------------
        :param X: Set of Data points
        :param Y: Set of Data Labels
        :param fnc: The submodular marginal gain function
        :param fwd_batch: The set of sampled points
        -------------------------------------------------
        """
        self.priority_queue = []
        super (LazyGreedy, self).__init__ (X, Y, batch_size)


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
        self.create_fwd_batch ()
        self.priority_queue = list (
            zip (list (map (lambda x: fnc (j, model, self. candidate_points,self.sample_points)), self.candidate_points)),
            self.candidate_points,
            [0] * len (self.candidate_points))

        heapq.heapify (self.priority_queue)
        temp = []
        while(len(temp)<self.cardinality):
            x = heapq.heappop (self.priority_queue)
            if x[2] == 0:
                alpha = fnc (x[1], model, self.sample_points)

            if x[2] == 1 or alpha > max (self.priority_queue, key=itemgetter (0))[0]:
                self.sample_points.append (x[1])
                temp.append(x[1])
                for i in self.priority_queue:
                    i = list (i)
                    i[2] = 0
                    i = tuple (i)
            else:
                heapq.heappush (self.priority_queue, (fnc (x[1], model, self.sample_points), x[1], 1))
        return temp


class ProbGreedy (Optimisation):

    def __init__(self, X, Y, batch_size):
        """
        -------------------------------------------------
        :param X: Set of Data points
        :param Y: Set of Data Labels
        :param fnc: The submodular marginal gain function
        :param fwd_batch: The set of sampled points
        -------------------------------------------------
        """

        super (ProbGreedy, self).__init__  (X, Y, batch_size)

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

        for i in range (0, self.cardinality):
            self.create_fwd_batch()
            prob_candidate_points = np.zeros(self.candidate_points.shape)
            for k, v in enumerate (self.candidate_points):
                prob_candidate_points[k] = fnc (v, model, self.candidate_points, self.sample_points)

            prob_candidate_points = self.softmax (prob_candidate_points)
            t = np.random.choice (self.candidate_points, size=1, p=prob_candidate_points)
            temp.extend(t)
            self.sample_points.extend(t)
        print(temp)
        return temp