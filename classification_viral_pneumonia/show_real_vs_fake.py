from keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt

C = {
    'size': (256, 256),
    'paths': {
        'Real X-ray': 'xrays_viral_pneumonia/Viral Pneumonia-1.png',
        'Well generated X-ray': 'generated_viral_pneumonia/0.png',
        'Poorly generated X-ray': 'generated_viral_pneumonia/4.png',
    },
    'color_mode': 'rgb'
}

images = [
    load_img(p, target_size=C['size'], color_mode=C['color_mode'])
    for p in C['paths'].values()
]

images = [img_to_array(i).astype('uint8') for i in images]

titles = (t for t in C['paths'])
fig, ax = plt.subplots(1, 3, figsize=(10, 10))
for a, t, i in zip(ax.flatten(), titles, images):
    a.axis('off')
    a.set_title(t)
    a.imshow(i)

fig.savefig('viral_pneumonia_real_fake.png')
plt.close(fig)
