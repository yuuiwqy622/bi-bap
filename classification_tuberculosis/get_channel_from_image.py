# Gets a particular color channel from each image in the source folder
# and outputs it to the destination folder. Created file has the same
# name as original file.

import argparse
import glob
from keras.preprocessing.image import load_img, img_to_array, save_img

parser = argparse.ArgumentParser(
    description='Extract channel'
)
parser.add_argument('--src_dir', type=str)
parser.add_argument('--dest_dir', type=str)
parser.add_argument('--size', type=int)

# Image file extension, for example jpg or png.
parser.add_argument('--format', type=str)

# Number of channel to extract. Select from 0 for red, 1 for green
# and 2 for blue.
parser.add_argument('--channel', type=int)
args = parser.parse_args()

SRC_DIR = args.src_dir
DEST_DIR = args.dest_dir
FORMAT = args.format
src_files = glob.glob(f'{SRC_DIR}/*.{FORMAT}')

for f in src_files:
    img = load_img(f, target_size=args.size)
    img = img_to_array(img)
    channel = img[:, :, args.channel:args.channel+1]

    fname = f.split('/')[-1]
    save_path = f'{DEST_DIR}/{fname}'
    save_img(save_path, channel)
    print(f'Channel saved to {save_path}')
