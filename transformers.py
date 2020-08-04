import silence_tensorflow.auto
import numpy as np
import random
import tensorflow.keras.preprocessing.image as keras_image
import errors as e
import tensorflow as tf
from errors import DimensionalityError
import gc
from imblearn.over_sampling import SMOTE
import pre_processors
from PIL import Image


class TransformerClassifier:
    """
        Abstract class that transforms any kind of data for any classifier (binary or multi-class)
    """

    def __init__(self, shuffle=False, smote_params=None):
        """
            Constructor of the class. To be implemented in Children classes
                :param shuffle:             (bool)          defines if the data will be randomized or not.
                :param smote_params:    (dict)      dictionary of integer parameters for the SMOTE algorithm to
                                                    augment training data. The keys of the dictionary are:
                                                    'random_sate' and 'k_neighbors'. Check scikit-learn library
         """
        self.shuffle = shuffle
        self.smote_params = smote_params

    @staticmethod
    def shuffle_data(x, y):
        """
            Shuffles the order of the training data and the labels.
                :param x:   (obj)       data to be augmented with shape: (n_samples, *dims)
                :param y:                   (np_array)  Array of labels to indicate to which class each image belongs to
                                                        Shape: (num_images,)
                :return:
        """
        if len(x) != len(y):
            raise DimensionalityError('Mismatch error between the number of training samples {},'
                                      ' and the number of labels {}.'.format(len(x), len(y)))

        random_idx = [i for i in range(len(x))]
        random.shuffle(random_idx)
        x = x[random_idx]
        y = y[random_idx]
        return x, y

    @staticmethod
    def smote(x, y, smote_params):
        """
                :param x:                   (obj)       data to be augmented with shape: (n_samples, *dims)
                :param y:                   (np_array)  Array of labels to indicate to which class each image belongs to
                                                        Shape: (num_images,)
                :param smote_params:        (dict)      dictionary of integer parameters for the SMOTE algorithm to
                                                        augment training data. The keys of the dictionary are:
                                                        'random_sate' and 'k_neighbors'. Check scikit-learn library
        """
        if len(x.shape) != 2:
            x = pre_processors.PreProcessor().flatten(x)

        # Ensure that the number of neighbors is less than or equal to the minimum class count
        min_class_count = np.min([np.count_nonzero(y == c) for c in np.unique(y)])
        if smote_params['k_neighbors'] >= min_class_count:
            smote_params['k_neighbors'] = min_class_count - 1

        if len(np.unique(y)) >= 2 and smote_params['k_neighbors'] > 0:
            sm = SMOTE(random_state=smote_params['random_state'], k_neighbors=smote_params['k_neighbors'])
            x, y = sm.fit_sample(x, y)
        return x, y

    def augment_data(self, x, y):
        """
            Method that augments all training data contained in the attribute "x".
            To be implemented in Children classes.
                :param x:       (obj)       data to be augmented with shape: (n_samples, *dims)
                :param y:       (np_array)  Array of labels to indicate to which class each image belongs to.
                                            Shape: (num_images,)
                :return: x: (np_array) augmented data.
        """
        if self.smote_params:
            x, y = self.smote(x, y, smote_params=self.smote_params)
        if self.shuffle:
            x, y = self.shuffle_data(x, y)
        return x, y


class ImageTransformerClassifier(TransformerClassifier):
    """
        Transformer object to manipulate and transform any kind of valid image data.
    """

    def __init__(self,
                 degree_range=10,
                 n_rotations=0,
                 horizontal_shift=0.05,
                 vertical_shift=0.05,
                 n_translations=0,
                 zoom_range=(0.85, 0.85),
                 n_zooms=0,
                 brightness_range=(0.0, 0.2),
                 n_brightness_shifts=0,
                 contrast_range=(0.8, 1.0),
                 n_contrast_shifts=0,
                 flip_left_right=False,
                 flip_up_down=False,
                 smote_params=None,
                 shuffle=True):
        """
            Constructor of the class.
                :param degree_range:        (int)           Rotation range, in degrees.
                :param n_rotations:         (int)           Number of random rotation iterations to perform.
                :param horizontal_shift:    (float)         Width shift range, as a float fraction of the width.
                :param vertical_shift:      (float)         Height shift range, as a float fraction of the height.
                :param n_translations:      (int)           Number of random translations to perform.
                :param zoom_range:          (float, float)  zoom range for width and height.
                :param n_zooms:             (int)           Number of random zooms to perform.
                :param brightness_range:    (float, float)  Levels of brightness: (low, high).
                :param n_brightness_shifts: (int)           Number of random brightness shifts to perform.
                :param contrast_range:      (float, float)  Levels of contrast: (low, high)
                :param n_contrast_shifts:   (int)           Number of random brightness shifts to perform.
                :param flip_left_right      (bool)          Flip images horizontally.
                :param flip_up_down:        (bool)          Flip images vertically.
                :param smote_params:        (dict)          dictionary of integer parameters for the SMOTE algorithm to
                                                            augment training data. The keys of the dictionary are:
                                                            'random_sate' and 'k_neighbors'. Check scikit-learn library.
                                                            It applies only to 1D classifiers.
                :param shuffle:             (bool)          defines if the data will be randomized or not.
                :return:                    None
        """
        super().__init__(shuffle=shuffle, smote_params=smote_params)
        self.degree_range = degree_range
        self.n_rotations = n_rotations
        self.horizontal_shift = horizontal_shift
        self.vertical_shift = vertical_shift
        self.n_translations = n_translations
        self.zoom_range = zoom_range
        self.n_zooms = n_zooms
        self.brightness_range = brightness_range
        self.n_brightness_shifts = n_brightness_shifts
        self.contrast_range = contrast_range
        self.n_contrast_shifts = n_contrast_shifts
        self.flip_left_right_flag = flip_left_right
        self.flip_up_down_flag = flip_up_down

    @staticmethod
    def rotate(images, degree_range=10, n_rotations=1):
        """
            Performs a random rotation of a Numpy image tensor.
            For more details check:
            https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_rotation

                :param images: 	        (np_array)  An array of gray scale images of shape
                                                    (num_images, img_width, img_height)
                :param degree_range:    (int)       Rotation range, in degrees.
                :param n_rotations:     (int)       Number of random rotation iterations to perform.
                :return:                (np_array)  Rotated Numpy image tensor.
        """
        if len(images.shape) != 3:
            raise e.DimensionalityError('Invalid shape {} for grayscale images array! Only shapes '
                                        '(num_images, img_width, img_height) are accepted.'.format(images.shape))

        if n_rotations <= 0:
            raise ValueError('Provide a positive number of rotations. Negative and 0 iterations are not allowed!')

        rotated = keras_image.random_rotation(images, rg=degree_range)
        for i in range(n_rotations - 1):
            rotated = np.append(rotated, keras_image.random_rotation(images, rg=degree_range),
                                axis=0)
        return rotated

    @staticmethod
    def translate(images, horizontal_shift=0.05, vertical_shift=0.05, n_translations=1):
        """
            Performs a random spatial shift of a Numpy image tensor.
            For more details check: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_shift

                :param images: 	            (np_array)  An array of gray scale images of shape
                                                        (num_images, img_width, img_height)
                :param horizontal_shift:    (float)     Width shift range, as a float fraction of the width.
                :param vertical_shift:      (float)     Height shift range, as a float fraction of the height.
                :param n_translations:      (int)       Number of random translations to perform.
                :return:                    (np_array)  Rotated Numpy image tensor.
        """
        if len(images.shape) != 3:
            raise e.DimensionalityError('Invalid shape {} for grayscale images array! Only shapes '
                                        '(num_images, img_width, img_height) are accepted.'.format(images.shape))

        if n_translations <= 0:
            raise ValueError('Provide a positive number of rotations. Negative and 0 iterations are not allowed!')

        translated = keras_image.random_shift(images, wrg=horizontal_shift, hrg=vertical_shift)
        for i in range(n_translations - 1):
            translated = np.append(translated,
                                   keras_image.random_shift(images, wrg=horizontal_shift, hrg=vertical_shift),
                                   axis=0)
        return translated

    @staticmethod
    def zoom(images, zoom_range=(0.85, 0.85), n_zooms=1):
        """
            Performs a random zoom of a Numpy image tensor.
            For more details check: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_zoom

                :param images: 	            (np_array)      An array of gray scale images of shape
                                                            (num_images, img_width, img_height)
                :param zoom_range:          (float, float)  zoom range for width and height.
                :param n_zooms:             (int)           Number of random zooms to perform.
                :return:                    (np_array)      Zoomed Numpy image tensor.
        """
        if len(images.shape) != 3:
            raise e.DimensionalityError('Invalid shape {} for grayscale images array! Only shapes '
                                        '(num_images, img_width, img_height) are accepted.'.format(images.shape))

        if n_zooms <= 0:
            raise ValueError('Provide a positive number of rotations. Negative and 0 iterations are not allowed!')

        zoomed = keras_image.random_zoom(images, zoom_range=zoom_range)
        for i in range(n_zooms - 1):
            zoomed = np.append(zoomed,
                               keras_image.random_zoom(images, zoom_range=zoom_range),
                               axis=0)
        return zoomed

    @staticmethod
    def adjust_brightness(images, brightness_range=(0, 0.2), n_shifts=1):
        """
            Adjust the brightness of RGB or Grayscale images.
            For more details check: https://www.tensorflow.org/api_docs/python/tf/image/adjust_brightness

                :param images: 	            (np_array)      An array of gray scale images of shape
                                                            (num_images, img_width, img_height)
                :param brightness_range:    (float, float)  Levels of brightness: (low, high).
                :param n_shifts:            (int)           Number of random brightness shifts to perform.
                :return:                    (np_array)      Numpy image tensor with new brightness.
        """
        if n_shifts <= 0:
            raise ValueError('Provide a positive number of rotations. Negative and 0 iterations are not allowed!')

        if brightness_range[0] > brightness_range[1]:
            raise ValueError('Incorrect tuple order for brightness range. '
                             'The first element in the tuple must be smaller than the second one.')

        if brightness_range[0] < 0:
            raise ValueError('Brightness level must be positive!')

        if brightness_range[0] >= 1:
            raise ValueError('Brightness level must be less than 1!')

        brightness = random.uniform(brightness_range[0], brightness_range[1])

        adjusted = tf.image.adjust_brightness(images, delta=brightness)
        for i in range(n_shifts - 1):
            adjusted = np.append(adjusted,
                                 tf.image.adjust_brightness(images, delta=brightness),
                                 axis=0)
        return adjusted

    @staticmethod
    def adjust_contrast(images, contrast_range=(0.8, 1), n_shifts=1):
        """
            Adjust the contrast of RGB or Grayscale images.
            For more details check: https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast

                :param images: 	            (np_array)      An array of gray scale images of shape
                                                            (num_images, img_width, img_height)
                :param contrast_range:      (float, float)  Levels of contrast: (low, high)
                :param n_shifts:            (int)           Number of random brightness shifts to perform.
                :return:                    (np_array)      Numpy image tensor with new contrast.
        """
        if n_shifts <= 0:
            raise ValueError('Provide a positive number of rotations. Negative and 0 iterations are not allowed!')

        if contrast_range[0] > contrast_range[1]:
            raise ValueError('Incorrect tuple order for contrast range. '
                             'The first element in the tuple must be smaller than the second one.')

        if contrast_range[0] < 0:
            raise ValueError('Contrast level must be positive!')

        if contrast_range[0] > 1:
            raise ValueError('Contrast level must be less than or equal than 1!')

        contrast = random.uniform(contrast_range[0], contrast_range[1])

        adjusted = tf.image.adjust_contrast(images, contrast_factor=contrast)
        for i in range(n_shifts - 1):
            adjusted = np.append(adjusted,
                                 tf.image.adjust_contrast(images, contrast_factor=contrast),
                                 axis=0)
        return adjusted

    @staticmethod
    def flip_left_right(images):
        """
            Flips the images from left to right.
                :param images:      (np_array)      An array of gray scale images of shape
                                                    (num_images, img_width, img_height)
                :return:            (np_array)      Dictionary of flipped Numpy image tensor
        """
        flipped = np.flip(images, axis=2)
        return flipped

    @staticmethod
    def flip_up_down(images):
        """
            Flips the images from top to bottom.
                :param images:      (np_array)      An array of gray scale images of shape
                                                    (num_images, img_width, img_height)
                :return:            (dict)          Dictionary of flipped Numpy image tensor.
        """
        flipped = np.flip(images, axis=1)
        return flipped

    def augment_data(self, images, labels):
        """
            Augment the numpy array of images with the specified conditions contained in the
            augmentation_params dictionary.
                :param images:  (np_array)  An array of images of shape
                                            (num_images, img_width, img_height, n_channels)
                :param labels:  (np_array)  Array of labels (int) to indicate to which class each image belongs to.
                                            Shape: (num_images,)
                :return: ((np_array), (np_array)) --> (augmented images, augmented labels).
        """

        if labels.shape != (images.shape[0],):
            raise DimensionalityError('Shape for labels array is inconsistent. Expected shape {}, '
                                      'but received shape {}'.format((images.shape[0],), labels.shape))

        original_shape = images.shape

        # If RGB image is passed convert it to grayscale
        if len(images.shape) == 4:
            images = tf.image.rgb_to_grayscale(images).numpy()
            images = np.squeeze(images, axis=3)

        # Detect classes
        class_list = list(set(list(labels)))

        # Initialize augmented training data array and its corresponding labels
        augmented = np.array([])
        augmented_labels = np.array([])

        for i, c in enumerate(class_list):

            # Extract images from class c from the training data
            images_c = images[labels == c]
            augmented_tmp = images_c

            if self.n_rotations:
                augmented_tmp = np.append(augmented_tmp,
                                          self.rotate(images_c,
                                                      degree_range=self.degree_range,
                                                      n_rotations=self.n_rotations),
                                          axis=0)

            if self.n_translations:
                augmented_tmp = np.append(augmented_tmp,
                                          self.translate(images_c,
                                                         horizontal_shift=self.horizontal_shift,
                                                         vertical_shift=self.vertical_shift,
                                                         n_translations=self.n_translations),
                                          axis=0)

            if self.n_zooms:
                augmented_tmp = np.append(augmented_tmp,
                                          self.zoom(images_c,
                                                    zoom_range=self.zoom_range,
                                                    n_zooms=self.n_zooms),
                                          axis=0)

            if self.flip_left_right_flag:
                augmented_tmp = np.append(augmented_tmp,
                                          self.flip_left_right(images=images_c),
                                          axis=0)

            if self.flip_up_down_flag:
                augmented_tmp = np.append(augmented_tmp,
                                          self.flip_up_down(images=images_c),
                                          axis=0)

            if self.n_contrast_shifts:
                augmented_tmp = np.append(augmented_tmp,
                                          self.adjust_contrast(images_c,
                                                               contrast_range=self.contrast_range,
                                                               n_shifts=self.n_contrast_shifts),
                                          axis=0)

            if self.n_brightness_shifts:
                augmented_tmp = np.append(augmented_tmp,
                                          self.adjust_brightness(images_c,
                                                                 brightness_range=self.brightness_range,
                                                                 n_shifts=self.n_brightness_shifts),
                                          axis=0)

            # Update class array with new augmented data
            if i == 0:
                augmented = augmented_tmp
            else:
                augmented = np.concatenate([augmented, augmented_tmp], axis=0)

            # Update labels
            augmented_labels = np.append(augmented_labels, np.full(shape=(augmented_tmp.shape[0],),
                                                                   fill_value=c))

            # Release unnecessary resources
            del augmented_tmp
            gc.collect()

            # Convert grayscale to RGB
            if len(original_shape) == 4:
                augmented = np.array([np.array(Image.fromarray(x_i).convert('RGB')) for x_i in augmented])

        if self.smote_params:
            augmented, augmented_labels = self.smote(augmented, augmented_labels, smote_params=self.smote_params)

        # Shuffle data
        if self.shuffle:
            augmented, augmented_labels = self.shuffle_data(augmented, augmented_labels)

        return augmented, augmented_labels
