import numpy as np
from keras import backend as K
from keras.datasets import mnist


def load_data():
    K.set_image_dim_ordering("th")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # normalization
    x_train = x_train[:, np.newaxis, :, :]
    return x_train, y_train, x_test, y_test
