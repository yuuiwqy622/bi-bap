import tensorflow as tf
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt

print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

IMAGE_DIR = 'dataset/image'
MASK_DIR = 'dataset/mask'
SAVE_DIR = 'dataset/concat_grayscale'
IMAGE_FILES = glob.glob(f'{IMAGE_DIR}/*.jpg')
MASK_FILES = glob.glob(f'{MASK_DIR}/*.png')
TARGET_SIZE = (256, 256)

IMAGE_FILES = sorted(IMAGE_FILES)
MASK_FILES = sorted(MASK_FILES)

with tf.device('/GPU:0'):
    for i, (image_f, mask_f) in enumerate(zip(IMAGE_FILES, MASK_FILES)):
        img = load_img(image_f, target_size=TARGET_SIZE, color_mode='grayscale')
        img = img_to_array(img)

        mask = load_img(mask_f, target_size=TARGET_SIZE, color_mode='grayscale')
        mask = img_to_array(mask)

        concat_img = np.concatenate([img, img, mask], axis=2)
        save_path = f'{SAVE_DIR}/{i}.png'
        save_img(save_path, concat_img)
        print(save_path)

