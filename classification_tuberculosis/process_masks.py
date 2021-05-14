import argparse
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, save_img

parser = argparse.ArgumentParser(
        description='process masks'
    )
parser.add_argument('--src_dir', type=str)
parser.add_argument('--dest_dir', type=str)
parser.add_argument('--format', type=str)
parser.add_argument('--size', type=int, default=256)
args = parser.parse_args()

SRC_DIR = args.src_dir
DEST_DIR = args.dest_dir

files = glob.glob(f'{SRC_DIR}/*{args.format}')

for f in files:
    mask = load_img(f, target_size=(args.size, args.size))
    mask = img_to_array(mask)
    mask[mask>=(255/2)]=255
    mask[mask<(255/2)]=0

    # fname = f.split('/')[-1]
    # save_path = f'{DEST_DIR}/{fname}'
    # save_img(save_path, mask)
    # print(f'Proccesed mask saved to {save_path}')