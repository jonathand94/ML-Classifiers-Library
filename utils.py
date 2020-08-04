import numpy as np
import gc
import random
from errors import DimensionalityError
import os
import pandas as pd
import pickle
import pydicom as dicom
from PIL import Image
import imageio
import tensorflow as tf


class FileManager:
    """
        Class that handles saving, writing and loading of files.
    """

    def __init__(self, file=None):
        """
            Creates an instance of the class.
                :param file:    Any type of file (currently supported: pickle files, CSV files).
        """
        self.file = file

    def load_file(self, file_path=''):
        """
            Method that loads the pickle object from the desired location and overrides the file attribute.
                :param  file_path   (str)           location where the pickle object is located.
                :return file        (pickle obj)    pickle object.
        """
        if not file_path.endswith('.pickle'):
            raise ValueError('Can only load Pickle files! Provide a valid Pickle extension!')
        self.file = pickle.load(open(file_path, 'rb'))
        return self.file

    def save_file(self, save_path=''):
        """
            Method that saves the pickle object to the desired location.
                :param  save_path   (str)   location where the data frame will be saved.
                :return None
        """
        if self.file is None:
            raise ValueError('A pickle object must be loaded before saving it! '
                             'Run the "load_file()" method first!')
        if not save_path.endswith('.pickle'):
            raise ValueError('Can only save Pickle files! Provide a valid Pickle extension!')

        # Get all folders
        list_sub_paths = save_path.split('/')

        # Save the pickle object if the path comprises just the file name
        if len(list_sub_paths) == 1:
            pickle.dump(self.file, open(save_path, 'wb'))
            return

        # Create folders, if the folders do not exist yet. Saves the file
        self.create_folders(save_path)
        pickle.dump(self.file, open(save_path, 'wb'))

    @staticmethod
    def extract_folder_path(file_path):
        """
            Returns the path of all folders from the provided file path.
                :param      file_path:  (str)   path where a file is located
                :return:    folder_path (str)   folder path where the file is located.
        """
        if '.' not in file_path:
            raise ValueError('The provided file path does not contain any extension!'
                             'Provide the extension of the file before exctracting the folder path!')
        list_sub_paths = file_path.split('/')
        return '/'.join(list_sub_paths[0:-1])

    def create_folders(self, file_path):
        """
            Method that creates all folders where the file will be located, if such folders do not exist.
                :param      file_path:  (str)   path where a file is located
                :return:    None
        """
        folder_path = self.extract_folder_path(file_path)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def get_classifier_image_training_data(training_data_path, shape=None, shuffle=True):
        """
            Method that returns a tuple (x, y), where the first element comprises a numpy array
            containing all image training data, which was collected from the provided "training_path".
            The second element comprises a numpy array with all labels assigned to the data.

            All folders inside the "training_path" will be considered as different classes for
            the classifier. Thus, the vector of labels "y" is created by detecting such folders.

            If different image sizes are encountered, the method provides a way to resize them
            to a common shape, which is either provided by the user in the argument "shape" or they
            are reshaped by a default shape of (128 x 128)

                :param training_data_path:   (str)   location where all data is located. All class instances must be divided
                                                in sub folders, for the method to create the corresponding class labels.
                :param shape:           (tuple) specifies the target shape to resize all image data.
                :param shuffle          (bool)  randomize training data.
                :return: tuple (x, y).
        """

        # Display classes detected
        classes = [c for c in os.listdir(training_data_path) if os.path.isdir(training_data_path + '/' + c)]
        classes.sort()
        msg = 'Number of classes {}:  {}'.format(len(classes), str([c for c in classes]))
        print(msg)

        # Create array with training data
        x = np.array([])
        y = np.array(())

        # Get valid image extensions
        valid_img_extensions = ['jpg', 'jpeg', 'png', 'bmp']

        for i, c in enumerate(classes):
            class_path = training_data_path + '/' + c

            # Create list of images from class i (only valid extensions are added)
            list_images = [img for img in os.listdir(class_path) if img.split('.')[-1] in valid_img_extensions]

            # Reshape all images and transform them to gray scale
            x_c = []
            for j, img in enumerate(list_images):
                print('\rGetting data from class "{}": {}/{}'.format(c, j, len(list_images)), end='')
                x_c.append(np.array(Image.open(class_path + '/' + img).convert('L').resize(shape)))
            x_c = np.array(x_c)

            # Create labels for class i
            y_c = np.empty(len(list_images))
            y_c.fill(i)

            # Concatenate x training data and y labels
            if i == 0:
                x = x_c
                y = y_c
            else:
                x = np.concatenate((x, x_c), axis=0)
                y = np.concatenate((y, y_c), axis=0)

            # Free memory
            del x_c, y_c
            gc.collect()

        if shuffle:
            idx = [i for i in range(len(x))]
            random.shuffle(idx)
            x = x[idx]
            y = y[idx]

        return x, y


class PandasDataFrameManager(FileManager):
    """
        Children class that handles saving, writing and loading of pandas Data Frames.
    """

    def __init__(self, df=None):
        """
            Creates an instance of the class.
                :param df   (Data Frame)    pandas Data Frame to be manipulated
        """
        super().__init__(file=df)

    def save_file(self, save_path=''):

        """
            Method that saves the data frame to the desired location.
                :param  save_path   (str)   location where the data frame will be saved.
                :return None
        """

        if self.file is None:
            raise ValueError('A pandas Data Frame must be loaded before saving it! '
                             'Run the "load_file()" method first!')
        elif type(self.file) is not pd.core.frame.DataFrame:
            raise TypeError('Only pandas Data Frames are accepted as arguments!')

        if not save_path:
            raise ValueError('Specify the location where the data frame will be saved!')
        elif not save_path.endswith('.csv'):
            raise ValueError('The save path must contain the .csv extension!')

        # Get all folders
        list_sub_paths = save_path.split('/')

        # Save the data frame if the path comprises just the file name
        if len(list_sub_paths) == 1:
            self.file.to_csv(save_path, index=False)
            return

        # Create folders, if the folders do not exist yet
        self.create_folders(save_path)
        self.file.to_csv(save_path, index=False)

    def save_multiple_df(self, save_path=''):

        """
            Saves multiple pandas Data Frames in an Excel file. Each Data Frame will correspond to a separate data
            sheet in the Excel file. Also, the dictionary keys correspond to the names of the seets in the Excel file.
                :param  save_path   (str)   location where the data frame will be saved.
                :return: None
        """

        if self.file is None:
            raise ValueError('A pandas Data Frame must be loaded before saving it! '
                             'Run the "load_file()" method first!')
        elif type(self.file) is not dict:
            raise TypeError('Only a dictionary of pandas Data Frames are accepted as arguments!')

        if not save_path:
            raise ValueError('Specify the location where the data frame will be saved!')
        if not save_path.endswith('.xlsx'):
            raise ValueError('The save path must contain the .xlsx extension!')

        writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
        for sheet_name, df in self.file.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()

    def load_file(self, file_path=''):
        """
            Method that loads the data frame from the desired location and overrides the file attribute.
            Returns the loaded file.
                :param  file_path   (str)           location where the CSV file is located.
                :return file        (Data Frame)    pandas Data Frame.
        """
        if not file_path.endswith('.csv'):
            raise ValueError('Can only load CSV files! Provide a valid CSV extension!')
        self.file = pd.read_csv(file_path, engine='python')
        return self.file

    def set_file(self, file):
        """
            Method that assigns a data frame to the file attribute of the class.
                :param file:  (Data Frame) pandas Data Frame to be assigned.
                :return:    None.
        """
        if file and type(file) is not pd.core.frame.DataFrame:
            raise TypeError('Only pandas Data Frames are accepted as arguments!')
        self.file = file

    @staticmethod
    def get_df_from_dict(dictionary, columns=None):
        """
            Returns a pandas DataFrame with two columns. One representing the keys
            and the second one representing the values of the dictionary.
                :param dictionary:  (dict)      dictionary to be transformed to a DataFrame
                :param columns:     (str, str)  list defining the column names for both keys and values
                                                of the dictionary.
                :return: pandas Data Frame
        """
        if not columns:
            columns = ['Keys', 'Values']
        keys = list(dictionary.keys())
        values = list(dictionary.values())
        data = [keys, values]
        df = pd.DataFrame(data=data).T
        df.columns = columns
        return df


class JsonFileManager(FileManager):
    """
        Children class that handles saving, writing and loading of json files.
    """

    def __init__(self, file=None):
        """
            Creates an instance of the class.
                :param file   (json serializable)   any json serializable object.
                                                    For more details check: https://pythontic.com/serialization/json/introduction
        """
        super().__init__(file=file)

    def save_file(self, save_path=''):
        """
            Method that saves the json object to the desired location.
                :param  save_path   (str)   location where the json file will be saved.
                :return None
        """
        if self.file is None:
            raise ValueError('A json serializable object must be loaded before saving it! '
                             'Run the "load_file()" method first!')
        if not save_path:
            raise ValueError('Specify the location where the json file will be saved!')
        if not save_path.endswith('.json'):
            raise ValueError('The save path must contain the .json extension!')

        # Get all folders
        list_sub_paths = save_path.split('/')

        # Save the data frame if the path comprises just the file name
        if len(list_sub_paths) == 1:
            with open(save_path, 'w') as json_file:
                json_file.write(self.file)
            return

        # Create folders, if the folders do not exist yet
        self.create_folders(save_path)
        with open(save_path, 'w') as json_file:
            json_file.write(self.file)

    def load_file(self, file_path=''):
        """
            Method that loads the json file from the desired location and overrides the file attribute.
            Returns the loaded file.
                :param  file_path   (str)           location where the json file is located.
                :return file        (json obj)      json object located in the specific path provided.
        """
        if not file_path.endswith('.json'):
            raise ValueError('Can only load json files! Provide a valid json extension!')

        json_file = open(file_path, 'r')
        self.file = json_file.read()
        json_file.close()
        return self.file


class DicomFileManager(FileManager):
    """
        Children class that handles saving, writing and loading of dicom files.
    """

    def __init__(self, file=None):
        """
            Creates an instance of the class.
                :param file   (dicom obj)   any dicom object.
                                            For more details check: https://pythontic.com/serialization/json/introduction
        """
        super().__init__(file=file)

    def get_pixel_data(self):
        """
            Method that extracts the pixel data from the dicom object.
                :return: dicom_pixels   (np_array) pixel data contained in the dicom object.
        """
        dicom_pixels = self.file.pixel_array
        return dicom_pixels

    def get_image_data(self):
        """
            Method that transforms all pixel data to image data.
                :return: image data contained in the dicom file.
        """
        # Get image data
        dicom_pixels = self.get_pixel_data()
        if len(dicom_pixels.shape) != 2:
            raise DimensionalityError('Cannot save an image from the dicom file, because it has an invalid shape {}. '
                                      'Only pixel data with shape (img_width, img_height) '
                                      'can be transformed to image data'.format(dicom_pixels.shape))

        dicom_pixels = dicom_pixels.astype(np.uint32)
        img = Image.fromarray(dicom_pixels, 'I')
        return img

    def save_pixels_as_image(self, save_path=''):
        """
            Method that saves the dicom object as an image to the desired location.
                :param  save_path   (str)   location where the dicom file will be saved.
                :return None
        """
        if self.file is None:
            raise ValueError('A dicom file must be loaded before saving it! '
                             'Run the "load_file()" method first!')
        if not save_path:
            raise ValueError('Specify the location where the dicom file will be saved!')
        valid_img_extensions = ['jpg', 'png', 'bmp', 'jpeg']
        if save_path.split('.')[-1] not in valid_img_extensions:
            raise ValueError('The save path must contain a valid image extension! '
                             'Valid extensions include: {}'.format(valid_img_extensions))

        # Get image data
        img = self.get_image_data()

        # Get all folders
        list_sub_paths = save_path.split('/')

        # Save the data frame if the path comprises just the file name
        if len(list_sub_paths) == 1:
            img.save(save_path)

        # Create folders, if the folders do not exist yet
        self.create_folders(save_path)

        try:
            img.save(save_path)
        except OSError:
            imageio.imwrite(save_path, self.get_pixel_data())

    def load_file(self, file_path=''):
        """
            Method that loads the dicom file from the desired location and overrides the file attribute.
            Returns the loaded file.
                :param  file_path   (str)           location where the dicom file is located.
                :return file        (dicom obj)     dicom object that was loaded.
        """
        if not file_path.endswith('.dcm'):
            raise ValueError('Can only load dicom files! Provide a valid dicom extension: ".dcm"!')

        self.file = dicom.dcmread(file_path)
        return self.file


class BinaryClassifierComparator:
    """
        Class that handles comparisons between different classifiers by performing statistical tests in them.
        The classifiers must be objects of type "Classifier"
    """

    def __init__(self, classifiers_dict, test_data_dict):
        """
            Constructor of the class that receives multiple Classifier objects, whose performance will
            be compared.
                :param classifiers_dict:    (dict)  dictionary of Classifier objects. Each key represent the ID
                                                    for the corresponding dictionary, while each value corresponds
                                                    to the classifier object.
                :param: test_data_dict:     (dict)  dictionary with all the test data to evaluate the binary
                                                    classifiers. The keys correspond to a single data instance ID
                                                    string, while the values represent a list of several two-size
                                                    tuples: (x, y).
        """
        self.classifiers_dict = classifiers_dict
        self.test_data_dict = test_data_dict
        self.predictions = {}
        self.comparison_df = None
        self.binary_metrics_df = None

        # TODO: add handling error if list do not contain a valid classifier

    def predict_samples(self, threshold=0.5):
        """"
            Method that computes all predictions for all data instances in the data dictionary,
            using all the classifiers. The method returns a dictionary where the keys represent
            the classifier IDs and each value defines a dictionary of predictions. Each classifier
            dictionary has the data instance IDs as keys and the values as the binary predicted labels.
                :param threshold:       [float]     threshold to be applied to predictions (either multi-class
                                                    or binary class)
                :return dictionary with all classifiers and their corresponding predictions.
        """
        for classifier_id, classifier in self.classifiers_dict.items():
            print('\rPredicting for classifier: {}'.format(classifier_id), end='')
            self.predictions[classifier_id] = {}
            for data_id, data_instance in self.test_data_dict.items():
                prediction = classifier.predict(x=data_instance[0], threshold=threshold)
                self.predictions[classifier_id][data_id] = prediction
                tf.keras.backend.clear_session()
        return self.predictions

    def compute_comparison_df(self):
        """
            Method that creates a pandas DataFrame, which combines all prediction results with the data labels.
        """
        df_data = {}
        df_data['Data ID'] = list(self.test_data_dict.keys())
        df_data['Labels'] = [int(self.test_data_dict[data_id][1]) for data_id in df_data['Data ID']]
        for classifier_id in self.classifiers_dict.keys():
            df_data[classifier_id] = [self.predictions[classifier_id][data_id][0][0] for data_id in df_data['Data ID']]
            df_data[classifier_id + ' result'] = np.logical_not(np.logical_xor(df_data['Labels'],
                                                                               df_data[classifier_id])).astype(int)
        self.comparison_df = pd.DataFrame(data=df_data)

    def get_comparison_df(self):
        """
            :return: pandas DataFrame with all predictions and data information.
        """
        if not self.comparison_df:
            self.compute_comparison_df()
        return self.comparison_df

    def compute_binary_metrics_df(self, x_test, y_test):
        """
            Method that creates a pandas DataFrame, which combines all prediction results with the data labels.

                :param x_test:              (np_array)  a matrix of feature variables with dimensions (mxn),
                                                        where n represents the number of feature variables
                                                        and m the number of validating examples
                :param y_test:              (np_array)  a vector of target variables with dimensions (mx1),
                                                        where m represents the number of validating target examples
        """
        # TODO: check resource exhaustion
        binary_metrics_data = {}
        binary_metrics = []
        for classifier_id, classifier in self.classifiers_dict.items():
            binary_metrics = classifier.get_binary_metrics(x_test=x_test, y_test=y_test)
            binary_metrics_data[classifier_id] = list(binary_metrics.values())
            tf.keras.backend.clear_session()
        binary_metrics_data['Binary Metric'] = list(binary_metrics.keys())
        self.binary_metrics_df = pd.DataFrame(data=binary_metrics_data)

    def get_binary_metrics_df(self, x_test, y_test):
        """
                :param x_test:              (np_array)  a matrix of feature variables with dimensions (mxn),
                                                        where n represents the number of feature variables
                                                        and m the number of validating examples
                :param y_test:              (np_array)  a vector of target variables with dimensions (mx1),
                                                        where m represents the number of validating target examples
                :return: pandas DataFrame with all binary metrics for all classifiers.
        """
        if not self.binary_metrics_df:
            self.compute_binary_metrics_df(x_test=x_test, y_test=y_test)
        return self.binary_metrics_df

    def perform_mcnemars_test(self, x):
        """
            Method that returns the McNemar's test result by comparing all classifiers in the attribute
            "list_classifiers" by using the data specified in the numpy array "x". Thus, all classifiers
            must admit the dimensions of matrix "x". This will help to select the best classifier in that list.
                :param x:   (np_array)  numpy array of data with shape: (n_samples, **dims)
                :return: mcnemar's test results.
        """
        # TODO: implement function

