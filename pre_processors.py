import silence_tensorflow.auto
from errors import DimensionalityError
import numpy as np
from utils import JsonFileManager
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import tensorflow.keras.models as models
import tensorflow.keras.applications.vgg16 as vgg16
import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.applications.inception_resnet_v2 as inception_resnet_v2
import tensorflow.keras.applications.densenet as densenet
import tensorflow.keras.applications.inception_v3 as inception_v3
from PIL import Image
import threading


class PreProcessor:
    """
        Class that performs multiple pre-processing tasks to the training data.
    """

    def __init__(self, n_pca_components=0, flatten_flag=False):
        """
            Constructor of the class.

            :param n_pca_components:    [int]   number of principal components that will be computed from the data "x"
                                                Must be an integer between 2 and 3.
            :param flatten_flag:        (bool)  whether to express the numpy array of data with shape
                                                [n_samples, n_features]
            :return: None.
        """
        self.n_pca_components = n_pca_components
        self.flatten_flag = flatten_flag

    @staticmethod
    def flatten(x):
        """
                :param x:   [np_array] matrix with all data of shape [n_samples, **dims]
                :return:    [np_array] matrix with shape [n_samples, n_features]
        """
        return np.reshape(x, (x.shape[0], np.product(x.shape[1::], )))

    def pca(self, x, n_pca_components=2):

        """
            Compute the principal components of the data matrix 'x'.

                :param x:                   [np_array] matrix with all data of shape [n_samples, n_features]
                :param n_pca_components:    [int] number of principal components that will be computed from the data "x"
                                            Must be an integer between 2 and 3.
                :return: numpy array of shape (n_samples, n_pca_components) with all principal components extracted.
        """

        if type(x) != np.ndarray:
            raise TypeError('TypeError: only numpy arrays are admitted for the matrix "x"!')
        if len(x.shape) > 2:
            x = self.flatten(x)
        elif len(x.shape) < 2:
            raise DimensionalityError('DimensionalityError: Invalid shape for the "x" matrix! A shape {0} was '
                                      'provided, but admissible shapes are: (n_samples, n_features)'.format(x.shape))
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=n_pca_components)
        return pca.fit_transform(x)

    def pre_process(self, x):
        """
            Computes all pre-processing steps indicated in the constructor of the PreProcessor object.
                :param x:   (np_array)  matrix with all data that will be visualized
                                        with shape [n_samples, n_features]
                :return:    (np_array)  pre-processed data.
        """
        if self.flatten_flag:
            return self.flatten(x)
        if self.n_pca_components:
            return self.pca(x, n_pca_components=self.n_pca_components)


class ImagePreProcessor(PreProcessor):

    """
        Children Class of PreProcessor. Used to transform image data before training.
    """

    def __init__(self,
                 flatten_flag=False,
                 n_pca_components=0,
                 extract_hog_flag=False,
                 type_cnn=''):
        """
            Constructor of the class.
                :param flatten_flag:        (bool)  whether to express the numpy array of data with shape
                                                    [n_samples, n_features]
                :param n_pca_components:    (int)   number of principal components that will be computed from the data
                                                    "x". Must be an integer between 2 and 3.
                :param type_cnn:            (str)   type of CNN to pre-process the data. Currently supported strings
                                                    are: 'inceptionv3', 'densenet', 'inception_resnet_v2', 'resnet50',
                                                    and 'vgg16'
                :param extract_hog_flag     (bool)  whether to extract hog features or not.
        """
        super().__init__(flatten_flag=flatten_flag, n_pca_components=n_pca_components)
        self.extract_hog_flag = extract_hog_flag
        self.type_cnn = type_cnn

    @staticmethod
    def extract_hog(x):

        """
            Extracts hog features from all image data contained in matrix "x"
                :param x:            [np_array] matrix with all image data with shape [n_samples, img_width, img_height]
                :return: hog_array:  [np_array] numpy array with hog features per slice [num slice, hog features]
        """

        if len(x.shape) != 3:
            raise DimensionalityError(
                'DimensionalityError: Invalid shape for the "x" image data matrix! A shape {0} was provided, '
                'but admissible shapes are: (n_samples, img_width, img_height)'.format(x.shape))
        hog_array = []
        for x_i in x:
            fd, hog_image = hog(x_i, orientations=8, pixels_per_cell=(8, 8),
                                cells_per_block=(1, 1), visualize=True, multichannel=False)
            hog_array.append(hog_image)
        return np.array(hog_array)

    @staticmethod
    def pre_process_built_cnn(x, type_cnn):
        """
            Method that pre-processes images based on the specified built convolutional neural network.
            Some of the supported CNNs include: VGG16, InceptionV3, DenseNet, ResNet50, and InceptionResNetV2
                :param x:           (np_array)  matrix with all image data with shape [n_samples, img_width, img_height]
                :param type_cnn:    (str)       type of CNN to pre-process the data. Currently supported strings are:
                                                'inceptionv3', 'densenet', 'inception_resnet_v2', 'resnet50',
                                                and 'vgg16'
                :return: (np_array)  pre-processed data.
        """

        valid_cnn_types = ['inceptionv3', 'densenet', 'inception_resnet_v2', 'resnet50', 'vgg16']
        if type_cnn not in valid_cnn_types:
            raise ValueError('You entered an invalid type of CNN: {}. '
                             'Valid CNN type arguments include: {}'.format(type_cnn, valid_cnn_types))

        if len(x.shape) < 3 or len(x.shape) > 4:
            raise DimensionalityError('Cannot pre-process image data with shape {} provided for the'
                                      'training data "x".\n'
                                      'Valid shapes are: (n_samples, img_width, img_height)'
                                      ' and (n_samples, img_width, img_height, n_channels)'.format(x.shape))

        # Transform gray scale data to RGB
        if len(x.shape) == 3:
            try:
                x = np.array([np.array(Image.fromarray(x_i).convert('RGB')) for x_i in x])
            except TypeError:
                x = np.array([np.array(Image.fromarray(np.uint8(x_i)).convert('RGB')) for x_i in x])

        if type_cnn == 'vgg16':
            x = vgg16.preprocess_input(x)
        elif type_cnn == 'resnet50':
            x = resnet50.preprocess_input(x)
        elif type_cnn == 'inception_resnet_v2':
            x = inception_resnet_v2.preprocess_input(x)
        elif type_cnn == 'densenet':
            x = densenet.preprocess_input(x)
        elif type_cnn == 'inceptionv3':
            x = inception_v3.preprocess_input(x)
        return x

    def pre_process(self, x):
        """
            Computes all pre-processing steps indicated in the constructor of the PreProcessor object.
                :param x:   (np_array)  matrix with all data that will be visualized
                                        with shape [n_samples, img_width, img_height]
                :return:    (np_array)  pre-processed data.
        """
        if self.type_cnn:
            x = self.pre_process_built_cnn(x, type_cnn=self.type_cnn)
        if self.extract_hog_flag:
            x = self.extract_hog(x)
        if self.flatten_flag:
            x = self.flatten(x)
        if self.n_pca_components:
            x = self.pca(x, n_pca_components=self.n_pca_components)
        return x


class XRayPreProcessor(ImagePreProcessor):

    """
        Children Class of ImagePreProcessor. Used to transform 2D X-Ray Images.
    """

    def __init__(self,
                 flatten_flag=False,
                 n_pca_components=0,
                 extract_hog_flag=False,
                 type_cnn=''):
        """
            Constructor of the class.
                :param flatten_flag:        (bool)  whether to express the numpy array of data with shape
                                                    [n_samples, n_features]
                :param n_pca_components:    (int)   number of principal components that will be computed from the data
                                                    "x". Must be an integer between 2 and 3.
                :param extract_hog_flag     (bool)  whether to extract hog features or not.
                :param type_cnn:            (str)   type of CNN to pre-process the data. Currently supported strings
                                                    are: 'inceptionv3', 'densenet', 'inception_resnet_v2', 'resnet50',
                                                    and 'vgg16'
        """
        super().__init__(flatten_flag=flatten_flag,
                         n_pca_components=n_pca_components,
                         extract_hog_flag=extract_hog_flag,
                         type_cnn=type_cnn)

    # TODO: implement methods


class ChestXRayPreProcessor(XRayPreProcessor):
    """
        Children Class of ImagePreProcessor. Used to transform 2D X-Ray Images only of the chest cavity.
    """

    def __init__(self,
                 flatten_flag=False,
                 n_pca_components=0,
                 extract_hog_flag=False,
                 type_cnn='',
                 segment_lungs_flag=False,
                 segment_lungs_method='',
                 u_net_model_path='',
                 u_net_weights_path='',
                 u_net_threshold=0.1,
                 u_net_batch_size=16,
                 verbose=0):
        """
            Constructor of the class.
                :param flatten_flag:        (bool)  whether to express the numpy array of data with shape
                                                    [n_samples, n_features]
                :param n_pca_components:    (int)   number of principal components that will be computed from the
                                                    data "x". Must be an integer between 2 and 3.
                :param extract_hog_flag     (bool)  whether to extract hog features or not.
                :param type_cnn:            (str)   type of CNN to pre-process the data. Currently supported strings
                                                    are: 'inceptionv3', 'densenet', 'inception_resnet_v2', 'resnet50',
                                                    and 'vgg16'
                :param segment_lungs_method (str)   strategy to segment lungs. Currently supported_ 'u_net'
                :param u_net_model_path     (str)   location where the trained model to segment the lungs is located.
                :param u_net_weights_path   (str)   location where the trained model weights is located.
                :param u_net_threshold      (float) Prediction threshold for u-net while segmenting.
                :param u_net_batch_size     (int)   Number of sample that U-Net will predict at each iteration.
                :param verbose              (int)   Amount of verbosity to show while predicting.
        """
        super().__init__(flatten_flag=flatten_flag,
                         n_pca_components=n_pca_components,
                         extract_hog_flag=extract_hog_flag,
                         type_cnn=type_cnn)
        self.segment_lungs_flag = segment_lungs_flag
        self.u_net_model_path = u_net_model_path
        self.u_net_weights_path = u_net_weights_path
        self.segment_lungs_method = segment_lungs_method
        self.u_net_threshold = u_net_threshold
        self.u_net_batch_size = u_net_batch_size
        self.verbose = verbose

    @staticmethod
    def segment_lungs(x,
                      model_path='',
                      weights_path='',
                      method='u_net',
                      batch_size=16,
                      threshold=0.5,
                      verbose=0):
        """
            Method that segments all X-Rays contained in the numpy array "x" of shape (n_samples, img_width, img_height)

                :param x:           (np_array)      array with shape (n_samples, img_width, img_height)
                                                    comprising all X-Rays to be segmented
                :param model_path   (str)           location where the trained model to segment the lungs is located.
                :param weights_path (str)           location where the trained model weights is located.
                :param method       (str)           strategy to segment lungs.
                :param batch_size   (int)           Number of samples per batch to segment.
                :param threshold    (float)         Prediction threshold for u-net while segmenting.
                :param verbose      (int)           Amount of verbosity to show while predicting.
                :return: x:         (np_array)      array with all segmented X-Rays.
        """

        # Initialize numpy array
        segmented = x

        if method == 'u_net':

            if threshold < 0 or threshold > 1:
                raise ValueError('Threshold for segmenting lungs must be between 0 and 1!')

            if not weights_path.endswith('.h5'):
                raise ValueError('Can only load h5 files! Provide a valid h5 extension!')

            # Expand dimensions of gray scale images
            if len(x.shape) == 3 and x.shape[1::] == (512, 512):
                x = np.expand_dims(x, axis=3)

            if x.shape[1::] != (512, 512, 1) or len(x.shape) != 4:
                raise DimensionalityError('Cannot segment lungs if the X-Rays have a shape different to: '
                                          '(n_x_rays, 512, 512, 1). A shape {} was provided instead'.format(x.shape))

            # Loading model
            lock = threading.Lock()
            lock.acquire()
            file_manager = JsonFileManager()
            model_json = file_manager.load_file(file_path=model_path)
            model = models.model_from_json(model_json)
            model.load_weights(weights_path)
            lock.release()

            # Getting predicted mask
            mask = model.predict(x, batch_size, verbose=verbose)

            # Match dimensions
            if len(x.shape) == 4:
                x = np.squeeze(x, axis=3)

            # Apply mask
            mask = np.squeeze(mask, axis=3)
            bool_mask = mask > threshold
            mask = bool_mask.astype(int)
            segmented = mask * x

        return segmented

    def pre_process(self, x):
        """
            Computes all pre-processing steps indicated in the constructor of the PreProcessor object.
                :param x:   (np_array)  matrix with all data that will be visualized
                                        with shape [n_samples, img_width, img_height]
                :return:    (np_array)  pre-processed data.
        """
        if self.segment_lungs_flag:
            x = self.segment_lungs(x,
                                   model_path=self.u_net_model_path,
                                   weights_path=self.u_net_weights_path,
                                   method=self.segment_lungs_method,
                                   threshold=self.u_net_threshold,
                                   batch_size=self.u_net_batch_size)
        if self.extract_hog_flag:
            x = self.extract_hog(x)
        if self.type_cnn:
            x = self.pre_process_built_cnn(x, type_cnn=self.type_cnn)
        if self.flatten_flag:
            x = self.flatten(x)
        if self.n_pca_components:
            x = self.pca(x, n_pca_components=self.n_pca_components)
        return x
