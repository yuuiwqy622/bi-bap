from keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt

C = {
    'size': (256, 256),
    'paths': (
        'xrays-x256/train/0/CHNCXR_0001_0.png',
        'xrays-x256/train/1/CHNCXR_0327_1.png',
        'fastgan/generated_healthy/0.png',
        'fastgan/generated_abnormal/0.png'
    ),
    'color_mode': 'rgb'
}

images = [
    load_img(p, target_size=C['size'], color_mode=C['color_mode'])
    for p in C['paths']
]

images = [img_to_array(i).astype('uint8') for i in images]

titles = 'Real normal', 'Real abnormal', 'Fake normal', 'Fake abnormal'
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for a, t, i in zip(ax.flatten(), titles, images):
    a.axis('off')
    a.set_title(t)
    a.imshow(i)

fig.savefig('xray_t_real_fake.png')
plt.close(fig)
