import tensorflow as tf
import glob
import argparse
from keras.preprocessing.image import load_img, img_to_array, save_img

parser = argparse.ArgumentParser(
        description='resize images'
    )
parser.add_argument('--format', type=str)
parser.add_argument('--src', type=str)
parser.add_argument('--dest', type=str)
args = parser.parse_args()

print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

FILES = glob.glob(f'{args.src}/*.{args.format}')
TARGET_SIZE = (256, 256)

with tf.device('/GPU:0'):
    for f in FILES:
        img = load_img(f, target_size = TARGET_SIZE)
        img = img_to_array(img)
        
        fname = f.split('/')[-1]
        save_name = f'{args.dest}/{fname}'
        print(save_name)
        save_img(save_name, img)