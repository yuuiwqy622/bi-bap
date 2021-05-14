from keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt

C = {
    'size': (256, 256),
    'paths': {
        'Real image': 'image/0.png',
        'Fake image 1': 'generated_image/3.png',
        'Fake image 2': 'generated_image/1276.png',
        'Real mask': 'mask/ISIC_0000000_segmentation.png',
        'Fake mask 1': 'generated_mask/3.png',
        'Fake mask 2': 'generated_mask/1276.png'
    },
    'color_mode': 'grayscale'
}

images = [
    load_img(p, target_size=C['size'], color_mode=C['color_mode'])
    for p in C['paths'].values()
]

images = [img_to_array(i).astype('uint8') for i in images]

titles = (t for t in C['paths'])
fig, ax = plt.subplots(2, 3, figsize=(10, 10))
for a, t, i in zip(ax.flatten(), titles, images):
    a.axis('off')
    a.set_title(t)
    a.imshow(i, cmap='gray')

fig.savefig('isic2018_real_fake.png')
plt.close(fig)
