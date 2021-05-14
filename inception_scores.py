import argparse
import glob
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Dropout, Reshape, Layer, Conv2DTranspose, Activation, BatchNormalization, LeakyReLU, AveragePooling2D, Flatten, Dense, ReLU
import tensorflow as tf
from itertools import product
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
import numpy as np
from math import ceil
import tensorflow.keras.backend as K
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray

image_generator = ImageDataGenerator()

BATCH_SIZE = 1

flow_args = {
    'batch_size': 1,
    'shuffle': False,
    'target_size': (299, 299),
    'class_mode': None,
    'color_mode': 'rgb',
    'seed': 1
}


parser = argparse.ArgumentParser(
        description='generate images'
    )
parser.add_argument('--image_dir', type=str)
parser.add_argument('--format', type=str)
args = parser.parse_args()

IMAGE_DIR = args.image_dir.split('/')
IMAGE_CLASS = IMAGE_DIR[-1]
IMAGE_DIR = '/'.join(IMAGE_DIR[:-1])

image_generator = image_generator.flow_from_directory(
    directory=IMAGE_DIR,
    classes=[IMAGE_CLASS],
    **flow_args
)


images = np.array([next(image_generator)[0] for _ in range(image_generator.samples)])

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, eps=1E-16):
    # load inception v3 model
    model = InceptionV3()
    # pre-process images, scale to [-1,1]
    images = preprocess_input(images)
    # predict p(y|x)
    p_yx = model.predict(images)
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # calculate KL divergence using log probabilities
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the log
    is_score = exp(avg_kl_d)
    # store
    # average across images
    return is_score

is_score = calculate_inception_score(images)
print(f'Inception scores: {is_score:.2f}')