from transformers import ImageTransformerClassifier
from visualizers import DataVisualizer
import numpy as np
import random


images = np.load('data/x_512.npy')
labels = np.load('data/y.npy')
start = random.randint(0, len(images) - 2)
num_images = 3

random_images = images[start:start + num_images]
random_labels = labels[start:start + num_images]

augmentation_params = {'degree_range': 10,
                       'n_rotations': 3,
                       'horizontal_shift': 0.05,
                       'vertical_shift': 0.05,
                       'n_translations': 3,
                       'zoom_range': (0.85, 0.85),
                       'n_zooms': 2,
                       'brightness_range': (0, 0.2),
                       'n_brightness_shifts': 2,
                       'contrast_range': (0.8, 1),
                       'n_contrast_shifts': 2,
                       'flip_left_right': True,
                       'flip_up_down': False,
                       'shuffle': True}

img_transformer = ImageTransformerClassifier(**augmentation_params)
augmented_images, augmented_labels = img_transformer.augment_data(images=random_images, labels=random_labels)

dv = DataVisualizer()
dv.plot_images(random_images, title='Original', image_names=random_labels)
dv.plot_images(augmented_images, title='Augmented', image_names=augmented_labels)
