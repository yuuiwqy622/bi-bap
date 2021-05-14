import tensorflow as tf
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt

print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

# Configuration
C = {
    'image_color_mode': 'grayscale',
    'mask_color_mode': 'rgb',
    'input_dir': 'fastgan_gray',
    'image_dir': 'images',
    'mask_dir': 'masks',
    'image_extension': 'jpg',
    'fig_save_dir': 'figures',
    'img_save_dir': 'combined'
}

IMAGE_FILES = glob.glob(f'{C["image_dir"]}/*.{C["image_extension"]}')
MASK_FILES = glob.glob(f'{C["mask_dir"]}/*.{C["image_extension"]}')
TARGET_SIZE = (256, 256)

IMAGE_FILES, MASK_FILES = sorted(IMAGE_FILES), sorted(MASK_FILES)

print(f'Loaded {len(IMAGE_FILES)} input images.')
print(f'Loaded {len(MASK_FILES)} mask images.')
    

with tf.device('/GPU:0'):
    for i, image_f, mask_f in zip(range(len(IMAGE_FILES)), IMAGE_FILES, MASK_FILES):
        img = load_img(
            image_f,
            target_size=TARGET_SIZE,
            color_mode=C['image_color_mode']
        )
        img = img_to_array(img).astype('uint8')

        mask = load_img(
            mask_f,
            target_size=TARGET_SIZE,
            color_mode=C['mask_color_mode']
        )
        mask = img_to_array(mask).astype('uint8')

        lungs, heart, trachea = mask[:,:,:1], mask[:,:,1:2], mask[:,:,2:]
        combined = np.concatenate([img, lungs, heart], axis=2)

        fig, ax = plt.subplots(1, 5, figsize=(10, 10))
        fig.suptitle(f'{image_f} : {mask_f}')

        for a, t in zip(ax, ['Image', 'Lungs', 'Heart', 'Trachea', 'Combined']):
            a.axis('off')
            a.set_title(t)
        
        ax[0].imshow(img, cmap='gray')
        ax[1].imshow(lungs, cmap='gray')
        ax[2].imshow(heart, cmap='gray')
        ax[3].imshow(trachea, cmap='gray')
        ax[4].imshow(combined)

        fig_save_path = f'{C["fig_save_dir"]}/{i}.png'
        fig.savefig(fig_save_path)
        plt.close(fig)

        img_save_path = f'{C["img_save_dir"]}/{i}.png'
        save_img(img_save_path, combined)
        print(img_save_path)
