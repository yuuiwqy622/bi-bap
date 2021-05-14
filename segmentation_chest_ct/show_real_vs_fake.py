from keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt
from itertools import chain

C = {
    'size': (256, 256),
    'paths': ('combined/2049.png', 'generated/110.png', 'generated/3.png'),
    'color_mode': 'rgb'
}

images = (
    load_img(
        p,
        target_size=C['size'],
        color_mode=C['color_mode']
    )
    for p in C['paths']
)

images = (img_to_array(i).astype('uint8') for i in images)
images = ((i[:, :, 0:1], i[:, :, 1:2], i[:, :, 2:]) for i in images)
images = chain.from_iterable(images)

titles = (f'{i} {j}' for i in ('Real', 'Generated', 'Generated')
          for j in ('CT scan', 'lungs mask', 'heart mask'))

fig, ax = plt.subplots(3, 3, figsize=(7, 7))
for a, t, i in zip(ax.flatten(), titles, images):
    a.axis('off')
    a.set_title(t)
    a.imshow(i, cmap='gray')

plt.subplots_adjust(
    wspace=1,
    hspace=0.2
)

fig.savefig('chest_ct_real_fake.png')
plt.close(fig)
