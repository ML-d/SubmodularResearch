import numpy as np
import time
import argparse
import sys
sys.path.append("/Users/kris/Desktop/ijcai2k18/code/")
from new_code import create_model
from new_code import read_data
from keras.models import Model


parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--dataset", type=str, default="mnist")
args = parser.parse_args ()

num_classes = 10
dataset = args.dataset
print(dataset)
x_train, y_train, x_test, y_test = read_data.read_data(dataset)
print(x_train.shape)

model = create_model.create_model(x_train.shape[1:], num_classes, "categorical_crossentropy", "mnist")
for i in range(0, 10):
    model.fit(x_train, y_train, batch_size = 512, shuffle=True)

intermediate_layer_model = Model (inputs=model.input, outputs=[model.get_layer ("prob").output,
                                           model.get_layer("features").output])
prob, features = intermediate_layer_model.predict(x_train)
filename = str(dataset)+"_feat.npy"
np.save(filename,features )