import time
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
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
import time
from PIL import Image
from glob import glob

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
parser.add_argument('--fake_dir', type=str)
parser.add_argument('--real_dir', type=str)
parser.add_argument('--color_mode', type=str)
args = parser.parse_args()

REAL_DIR = args.real_dir.split('/')
REAL_CLASS = REAL_DIR[-1]
REAL_DIR = '/'.join(REAL_DIR[:-1])

FAKE_DIR = args.fake_dir.split('/')
FAKE_CLASS = FAKE_DIR[-1]
FAKE_DIR = '/'.join(FAKE_DIR[:-1])

# assumes images have any shape and pixels in [0,255]
def calculate_fid(model, images1, images2):
  # calculate activations
  act1 = model.predict(images1)
  act2 = model.predict(images2)
  # calculate mean and covariance statistics
  mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
  # calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2)**2.0)
  # calculate sqrt of product between cov
  covmean = sqrtm(sigma1.dot(sigma2))
  # check and correct imaginary numbers from sqrt
  if iscomplexobj(covmean):
   covmean = covmean.real
  # calculate score
  fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid

image_generator = ImageDataGenerator()

BATCH_SIZE = 1

flow_args = {
    'batch_size': 1,
    'shuffle': False,
    'target_size': (299, 299),
    'class_mode': None,
    'color_mode': args.color_mode,
    'seed': 1
}

image_generator1 = image_generator.flow_from_directory(
    directory=FAKE_DIR,
    classes=[FAKE_CLASS],
    **flow_args
)

image_generator2 = image_generator.flow_from_directory(
    directory=REAL_DIR,
    classes=[REAL_CLASS],
    **flow_args
)

start = time.time()

real = np.array([next(image_generator1)[0] for i in range(image_generator2.samples)])
fake = np.array([next(image_generator2)[0] for i in range(image_generator1.samples)])
print('Loaded batches:', fake.shape, real.shape)

model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

real = preprocess_input(real)
fake = preprocess_input(fake)

fid = calculate_fid(model, real, fake)

end = time.time()
print(f'Duration: {end-start} s')

print('FID: %.2f' % fid)