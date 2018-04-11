import keras
import numpy as np
import keras.backend as K
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--dataset", type = str, default="mnist")
parser.add_argument("--distance", type = str, choices=["l2", "cosine"])
args = parser.parse_args()
dataset = args.dataset
file_name = dataset+"_feat.npy"
data_feat = np.load(dataset)
data_feat = normalize(data_feat, axis=1)
if args.distance == "cosine":
    dist_matrix = np.dot(data_feat, np.transpose(data_feat))
if args.distance == "l2":
    dist_matrix = np.zeros((data_feat.shape[0], data_feat.shape[0]))
    for i in range(0, data_feat.shape[0]):
        dist_matrix[i] = np.linalg.norm(data_feat[i] - data_feat, axis=1)


np.save(str(dataset)+str(args.distance)+".npy", dist_matrix)



