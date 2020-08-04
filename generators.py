import silence_tensorflow.auto
from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image
from errors import *
import time
import pre_processors
import transformers
from visualizers import DataVisualizer
import random


class ClassifierGenerator(Sequence):
    """
        Abstract class for any generator applied to classifiers.
    """
    def __init__(self):
        """
            Constructor of the class. To be implemented in Children classes.
        """

    def __len__(self):

        """
            Denotes the number of training batches (iterations) per epoch.
                :return: The number of iterations that the model will train until all the data was used.
        """

    def on_epoch_end(self):
        """
            Updates indexes after each epoch
                :return: None.
        """

    def __getitem__(self, index):

        """
            Generate one batch of data
                :param index:   (int)       starting index of the training array 'x' to start collecting the
                                            corresponding batch of data.
                :return: (x, y) (tuple)     tuple, where the first element corresponds to the training data "x" of
                                            the batch and the second element to the respective labels.
        """

    def __data_generation(self, x_file_paths_tmp):
        """
            Generates data containing batch_size samples
                :param x_file_paths_tmp:        (list)  list of file paths with extensions, indicating the corresponding
                                                        locations of the samples that will be extracted. Extensions that
                                                        are currently supported include: 'jpg', 'jpeg', 'png', 'bmp'.
                                                        (i.e. ["file_path_1", "file_path_2", ...])
                :return: The batch of data.
        """


class ClassifierImageGenerator(ClassifierGenerator):

    """
        Class that generates data for any classifier object. It inherits methods from
        the ClassifierDataGenerator class.
    """

    def __init__(self,
                 x_file_paths,
                 labels_dict,
                 input_shape=None,
                 image_shape=(512, 512),
                 batch_size=32,
                 shuffle=True,
                 transformer=transformers.ImageTransformerClassifier(),
                 pre_processor=pre_processors.ImagePreProcessor(),
                 verbose=2):

        """
            Constructor of the class.
                :param x_file_paths:        (list)              list of file paths with extensions, indicating the
                                                                corresponding locations of the samples that will be
                                                                extracted. (i.e. ["file_path_1", "file_path_2", ...])
                :param labels_dict:         (dict)              dictionary that defines the labels associated to each
                                                                training sample. (i.e. {"file_path_1": 1, ...})
                :param input_shape:         (tuple)             input shape of the training data for the classifier:
                                                                (n_samples, **dims)
                :parm image_shape:          (tuple)             target shape of the image data that will be generated
                                                                (n_samples, img_width, img_height, n_channels),
                                                                (n_samples, img_width, img_height)
                :param batch_size:          (int)               size of the batch to generate at each iteration of the
                                                                data generation.
                :param shuffle:             (bool)              defines if the data will be randomized or not.
                :param transformer:         (ImageTransformer)  transformer object that will augment the data. Check the
                                                                documentation in the transformers module.
                                                                Check scikit-learn library documentation for details:
                :param pre_processor        (ImagePreProcessor) pre-processor object that will pre-process the data.
                                                                Check the documentation in the utils module.
                :param verbose:             (int)               how much information will be displayed when training.
                                                                [Valid arguments: '0', '1' or '2']
                :return None
        """

        if type(transformer) != transformers.ImageTransformerClassifier and \
                type(transformer) != transformers.TransformerClassifier:
            raise TypeError('Cannot use image data generator with a transformer that is not'
                            ' of the type ImageTransformer or TransformerClassifier. '
                            'See the transformers module for details.')

        if type(pre_processor) != pre_processors.ImagePreProcessor and \
                type(pre_processor) != pre_processors.XRayPreProcessor and \
                type(pre_processor) != pre_processors.ChestXRayPreProcessor:
            raise TypeError('Cannot use image data generator with a pre-processor that is not'
                            ' of the following types: "ImagePreProcessor", "XRayPreProcessor"'
                            ' or "ChestXRayPreProcessor". See the pre_processors module for details.')

        # Initialization
        super().__init__()
        self.input_shape = input_shape
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.labels_dict = labels_dict
        self.x_file_paths = x_file_paths
        self.shuffle = shuffle
        self.transformer = transformer
        self.pre_processor = pre_processor
        self.verbose = verbose

        # Specify data formats and indices
        self.indexes = np.arange(len(self.x_file_paths))
        self.num_batches = 0
        self.current_batch = 0
        self.n_epoch = 1
        self.on_epoch_end()

        # Get progress bar to update data generation
        self.dv = DataVisualizer(x=None)

    def __len__(self):

        """
            Denotes the number of training batches (iterations) per epoch.
                :return: The number of iterations that the model will train until all the data was used.
        """
        self.num_batches = int(np.floor(len(self.x_file_paths) / self.batch_size))
        return self.num_batches

    def on_epoch_end(self):
        """
            Updates indexes after each epoch
                :return: None.
        """
        self.current_batch = 0
        self.indexes = np.arange(len(self.x_file_paths))
        if self.shuffle:
            random_idx = [i for i in range(len(self.x_file_paths))]
            random.shuffle(random_idx)
            self.x_file_paths = list(np.array(self.x_file_paths)[random_idx])
        self.n_epoch += 1
        if self.verbose == 0 or self.verbose > 2:
            print()

    def __data_generation(self, x_file_paths_tmp):
        """
            Generates data containing batch_size samples
                :param x_file_paths_tmp:        (list)  list of file paths with extensions, indicating the corresponding
                                                        locations of the samples that will be extracted. Extensions that
                                                        are currently supported include: 'jpg', 'jpeg', 'png', 'bmp'.
                                                        (i.e. ["file_path_1", "file_path_2", ...])
                :return: The batch of data.
        """

        # Initialize arrays
        x = []
        y = []

        # Iterate over all file paths and create Image array
        for i, img_path in enumerate(x_file_paths_tmp):
            x.append(np.array(Image.open(img_path).convert('L').resize(self.image_shape)))
            y.append(self.labels_dict[img_path])

        x = np.array(x)
        y = np.array(y, dtype=int)

        if self.transformer:
            x, y = self.transformer.augment_data(x, y)

        if self.pre_processor:
            x = self.pre_processor.pre_process(x)

        # Convert Grayscale to RGB
        if len(self.input_shape) == 4 and len(x.shape) == 3:
            x = np.array([np.array(Image.fromarray(img).convert('RGB')) for img in x])

        # For NN or other 1D classifiers
        if len(self.input_shape) == 2 and len(x.shape) != 2:
            x = np.reshape(x, (x.shape[0], np.product(x.shape[1::],)))

        return x, y

    def __getitem__(self, index):

        """
            Generate one batch of data
                :param index:   (int)       starting index of the training array 'x' to start collecting the
                                            corresponding batch of data.
                :return: (x, y) (tuple)     tuple, where the first element corresponds to the training data "x" of
                                            the batch and the second element to the respective labels.
        """

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        x_file_paths_tmp = [self.x_file_paths[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(x_file_paths_tmp)

        # Check if flatten is required based on the desired input shape
        if len(self.input_shape) == 1:
            x = np.reshape(x, (x.shape[0], np.product(x.shape[1::])))

        if self.verbose == 0 or self.verbose == 2 and self.n_epoch != 1:
            # UNCOMMENT IF NOT GOOGLE COLAB
            # self.dv.print_progress_bar(self.current_batch,
            #                            self.__len__(),
            #                            prefix='Generating batches of shape {}'.format(x.shape),
            #                            suffix='',
            #                            length=20)
            print('\rGenerating batch {}/{} of shape {}'.format(self.current_batch,
                                                                self.__len__(),
                                                                x.shape), end='')

        # New batch to train with
        self.current_batch += 1

        return x, y
