import silence_tensorflow.auto
from sklearn.model_selection import StratifiedKFold
from utils import *
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
import tensorflow.keras.metrics as km
from tensorflow.keras.models import model_from_json
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.layers import Dense, Input, Flatten
import random
from visualizers import DataVisualizer
from errors import *
import custom_km
import pandas as pd
from contextlib import redirect_stdout
import warnings
import sklearn.metrics as skm
import tensorflow.keras.models as models
import tensorflow.keras.applications.vgg16 as vgg16
import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.applications.inception_resnet_v2 as inception_resnet_v2
import tensorflow.keras.applications.densenet as densenet
import tensorflow.keras.applications.inception_v3 as inception_v3
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import pre_processors
import transformers
import generators
from copy import deepcopy
import threading


class Classifier:
    """
        Abstract generalization of a supervised learning classifier.
    """

    def __init__(self, n_classes=2, metrics=None, input_shape=None):
        """
            Constructor of the class.
                :param n_classes:   (int)   number of classes to be classified.
                :param input_shape:         (tuple) input shape of the training data
                :param metrics:     (list)  could be a list of strings or a list of custom Keras metrics, depending
                                            on the children class. Valid metrics include: F1-score, sensitivity,
                                            specificity, AUC, precision, and accuracy.
        """

        if not n_classes >= 2:
            raise ValueError('ValueError: Specify a valid number of classes. '
                             'It must be an integer greater than or equal to 2')

        self.n_classes = n_classes
        self.metrics = metrics
        self.input_shape = input_shape
        self.type_classifier = 'classifier'
        self.type_generator = ''

        # Display format
        self.verbose = 2

        if self.n_classes == 2:
            self.loss = 'binary_crossentropy'
        else:
            self.loss = 'categorical_crossentropy'

        # Model attributes
        self.model = None
        self.models_history = {}
        self.best_model = None
        self.trained_models = []

        # Data Attributes
        self.target_metric = 'f1'
        self.x_original_shape = self.input_shape
        self.confusion_matrix = np.array([])

        # Hyper parameter attributes
        self.hyper_parameter_space = {}
        self.hyper_parameters = {}
        self.best_hyper_parameters = {}

        # Transformer object
        self.transformer = transformers.TransformerClassifier()

        # PreProcessor object
        self.pre_processor = pre_processors.PreProcessor()

    def predict_proba(self, x, use_best_model=False):

        """
            Method that predicts the labels based on the corresponding matrix of data 'x'.
                :param x:               [np_array]  matrix of training data
                :param use_best_model:  [bool]      whether to use the best model to make predictions
                                                    or use the last trained model
                :return y:              [np_array]  predicted labels.
        """

        if type(x) != np.ndarray:
            raise ValueError('"x" data must be numpy array!')

        model = self.model
        if use_best_model:
            model = self.best_model

        if model is None:
            raise ModelError('Cannot predict samples because no model has been constructed! '
                             'Please build a model before predicting')

        x = self.pre_processor.pre_process(x)

        # Check that the  input training data has the same shape
        # as the data that was trained before applying pre-processing techniques
        if x.shape[1::] != self.x_original_shape[1::]:
            raise DimensionalityError('Invalid shape of {} for pre_processed "x" data numpy array! '
                                      'Data needs to have the following shape: {}'.format(x.shape[1::],
                                                                                          self.x_original_shape[1::]))
        return model.predict_proba(x)

    def predict(self, x, use_best_model=False, threshold=0.5):

        """
            Method that predicts the labels based on the corresponding matrix of data 'x'.
                :param x:               [np_array]  matrix of training data
                :param use_best_model:  [bool]      whether to use the best model to make predictions
                                                    or use the last trained model
                :param threshold:       [float]     threshold to be applied to predictions (either multi-class
                                                    or binary class)
                :return y:              [np_array]  predicted labels.
        """
        probabilities = self.predict_proba(x=x, use_best_model=use_best_model)
        return (probabilities >= threshold).astype(np.int)

    def get_confusion_matrix(self, x_test, y_test):

        """
            Method to calculates the confusion matrix of the binary classifier.
                :param x_test:              (np_array)  a matrix of feature variables with dimensions (mxn),
                                                        where n represents the number of feature variables
                                                        and m the number of validating examples
                :param y_test:              (np_array)  a vector of target variables with dimensions (mx1),
                                                        where m represents the number of validating target examples
                :return: confusion_matrix   (np_array)  Confusion matrix with shape (n_classes, n_classes) whose i-th
                                                        row and j-th column entry indicates the number of samples with
                                                        true label being i-th class and prediced label being j-th class.
        """

        if self.n_classes != 2:
            raise ValueError('Cannot compute the confusion matrix of a classifier that is not binary. '
                             'Build a classifier with an attribute "n_classes"=2 for computing the'
                             'confusion matrix.')

        # Find the predictions of the model
        y_pred = self.predict(x_test)

        # Find the confusion matrix
        self.confusion_matrix = skm.confusion_matrix(y_test, y_pred)
        return self.confusion_matrix

    def get_tn(self, x_test, y_test):
        """
            Method to calculates the confusion matrix of the binary classifier.
                :param x_test:              (np_array)  a matrix of feature variables with dimensions (mxn),
                                                        where n represents the number of feature variables
                                                        and m the number of validating examples
                :param y_test:              (np_array)  a vector of target variables with dimensions (mx1),
                                                        where m represents the number of validating target examples
                :return: tn:                (int)       number of negative correct instance predictions.
        """
        tn, _, _, _ = self.get_confusion_matrix(x_test=x_test, y_test=y_test)
        return tn

    def get_fp(self, x_test, y_test):
        """
            Method to calculates the confusion matrix of the binary classifier.
                :param x_test:              (np_array)  a matrix of feature variables with dimensions (mxn),
                                                        where n represents the number of feature variables
                                                        and m the number of validating examples
                :param y_test:              (np_array)  a vector of target variables with dimensions (mx1),
                                                        where m represents the number of validating target examples
                :return: fp:                (int)       number of positive incorrect instance predictions.
        """
        _, fp, _, _ = self.get_confusion_matrix(x_test=x_test, y_test=y_test)
        return fp

    def get_fn(self, x_test, y_test):
        """
            Method to calculates the confusion matrix of the binary classifier.
                :param x_test:              (np_array)  a matrix of feature variables with dimensions (mxn),
                                                        where n represents the number of feature variables
                                                        and m the number of validating examples
                :param y_test:              (np_array)  a vector of target variables with dimensions (mx1),
                                                        where m represents the number of validating target examples
                :return: fn:                (int)       number of negative incorrect instance predictions.
        """
        _, _, fn, _ = self.get_confusion_matrix(x_test=x_test, y_test=y_test)
        return fn

    def get_tp(self, x_test, y_test):
        """
            Method to calculates the confusion matrix of the binary classifier.
                :param x_test:              (np_array)  a matrix of feature variables with dimensions (mxn),
                                                        where n represents the number of feature variables
                                                        and m the number of validating examples
                :param y_test:              (np_array)  a vector of target variables with dimensions (mx1),
                                                        where m represents the number of validating target examples
                :return: tp:                (int)       number of positive correct instance predictions.
        """
        _, _, _, tp = self.get_confusion_matrix(x_test=x_test, y_test=y_test)
        return tp

    def get_confusion__matrix_fig(self, x_test, y_test):

        """
            Method to visualize the confusion matrix of the binary classifier.
                :param x_test:              (np_array)      a matrix of feature variables with dimensions (mxn),
                                                            where n represents the number of feature variables
                                                            and m the number of validating examples
                :param y_test:              (np_array)      a vector of target variables with dimensions (mx1),
                                                            where m represents the number of validating target examples
                :return:                    (plotly Figure) plotly Figure representing the confusion matrix.
        """

        tn, fp, fn, tp = self.get_confusion_matrix(x_test=x_test, y_test=y_test).ravel()

        values = [['<b>TRUTH</b> <b>0</b>', '<b>TRUTH</b> <b>1</b>'], [tn, fn], [fp, tp]]

        data = [go.Table(columnorder=[1, 2, 3],
                         columnwidth=[100, 100, 100],
                         header=dict(
                             values=[[''], ['<b>PREDICTED</b> <b>0</b>'], ['<b>PREDICTED</b> <b>1</b>']],
                             line_color='darkslategray',
                             fill_color='royalblue',
                             align=['center', 'center', 'center'],
                             font=dict(color='white', size=12),
                             height=40),
                         cells=dict(
                             values=values,
                             line_color='darkslategray',
                             fill=dict(color=['royalblue', 'white', 'white']),
                             align=['center', 'center', 'center'],
                             font=dict(color=['white', 'black', 'black'], size=12),
                             height=30)
                         )
                ]
        return go.Figure(data=data)

    def get_binary_metrics(self, x_test, y_test):
        """
            Returns a dictionary with all binary metrics computed. Applies only for binary classifiers.

                :param x_test:              (np_array)  a matrix of feature variables with dimensions (mxn),
                                                        where n represents the number of feature variables
                                                        and m the number of validating examples
                :param y_test:              (np_array)  a vector of target variables with dimensions (mx1),
                                                        where m represents the number of validating target examples
                :return:
        """

        binary_metrics = {}
        tn, fp, fn, tp = self.get_confusion_matrix(x_test=x_test, y_test=y_test).ravel()

        # Find the predictions of the model
        y_proba = self.predict_proba(x_test)

        # Update dictionary
        binary_metrics['tn'] = tn
        binary_metrics['fp'] = fp
        binary_metrics['fn'] = fn
        binary_metrics['tp'] = tp

        # Calculate accuracy, precision, sensitivity, specificity and f1 scores
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
        auc = skm.roc_auc_score(y_test, y_proba)

        # Update dictionary
        binary_metrics['accuracy'] = accuracy
        binary_metrics['precision'] = precision
        binary_metrics['sensitivity'] = sensitivity
        binary_metrics['specificity'] = specificity
        binary_metrics['f1'] = f1
        binary_metrics['auc'] = auc
        return binary_metrics

    def get_roc_curve_fig(self, x_test, y_test):

        """
            Method that allows the visualization of the ROC Curve, AUC and best threshold.

                :param x_test:  (np_array)              a matrix of feature variables with dimensions (mxn),
                                                        where n represents the number of feature variables
                                                        and m the number of validating examples
                :param y_test:  (np_array)              a vector of target variables with dimensions (mx1),
                                                        where m represents the number of validating target examples
                :return:        (Matplotlib figure)     figure containing the ROC curve and the optimal threshold.
        """

        if self.n_classes != 2:
            raise ValueError('Cannot compute the confusion matrix of a classifier that is not binary. '
                             'Build a classifier with an attribute "n_classes"=2 for computing the'
                             'confusion matrix.')

        # Calculate the probability
        y_proba = self.predict_proba(x_test)

        # Calculate false positive rate, true positive rate and thresholds
        fpr, tpr, thresholds = skm.roc_curve(y_test, y_proba, drop_intermediate=False)

        # Find best threshold
        optimal_threshold_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_threshold_idx]
        print('\nOptimal threshold:', optimal_threshold)

        # Calculate AUC
        auc = skm.roc_auc_score(y_test, y_proba)

        # Plot roc curve
        fig = plt.figure()
        plt.plot(fpr, tpr, "r", linewidth=3, label='AUC {}'.format('%.4f' % auc))
        plt.scatter((fpr[optimal_threshold_idx]), tpr[optimal_threshold_idx], alpha=1, color='b',
                    label='Best Threshold {}'.format('%.4f' % thresholds[optimal_threshold_idx]))
        plt.title('ROC Curve', fontweight="bold", fontsize=12)
        plt.xlabel('False Positive Rate', fontweight="bold")
        plt.ylabel('True Positive Rate', fontweight="bold")
        plt.legend()
        return fig

    def get_training_parameters_df(self):
        """
            Method that returns a pandas Data Frame with all parameters that were used for training.
                :return: (Data Frame)
        """

        # Get metrics
        metrics = []
        for metric in self.metrics:
            if type(metric) == str:
                metrics.append(metric)
            else:
                metrics.append(metric.name)

        # Create training dictionary
        training_parameters_dict = {'x original shape': str(self.x_original_shape),
                                    'generator': self.type_generator,
                                    'x shape': str(self.input_shape),
                                    'loss': str(self.loss),
                                    'classes': str(self.n_classes),
                                    'metrics': str(metrics)}

        training_parameters = list(training_parameters_dict.keys())
        values = list(training_parameters_dict.values())

        data = {'Training Parameters': training_parameters, 'Values': values}
        df = pd.DataFrame(data=data)
        return df

    def get_pre_processing_parameters_df(self):
        """
            Method that returns a pandas Data Frame with all parameters that were used for training.
                :return: (Data Frame)
        """
        df_manager = PandasDataFrameManager()
        return df_manager.get_df_from_dict(dictionary=self.pre_processor.__dict__, columns=['Parameter', 'Value'])

    def get_augmentation_parameters_df(self):
        """
            Method that returns a pandas Data Frame with all parameters that were used for training.
                :return: (Data Frame)
        """
        df_manager = PandasDataFrameManager()
        return df_manager.get_df_from_dict(dictionary=self.transformer.__dict__, columns=['Parameter', 'Value'])

    def get_hyper_parameters_df(self):
        """
            Method that returns a pandas Data Frame with the hyper parameter values retrieved from the model.
                :return: (Data Frame)
        """
        if not self.hyper_parameters:
            raise HyperParameterError('Cannot get the best hyper parameter Data Frame '
                                      'if no model has been constructed.')

        hyper_parameters = [str(h) for h in list(self.hyper_parameters.keys())]
        values = [str(v) for v in self.hyper_parameters.values()]
        data = {'Hyper Parameters': hyper_parameters, 'Values': values}
        df = pd.DataFrame(data=data)
        return df

    def get_best_hyper_parameters_df(self):
        """
            Method that returns a pandas Data Frame with the best hyper parameter values retrieved from the best model.
                :return: (Data Frame)
        """
        if not self.best_hyper_parameters:
            raise HyperParameterError('Cannot get the best hyper parameter Data Frame '
                                      'if no model has been constructed.')

        hyper_parameters = [str(h) for h in list(self.best_hyper_parameters.keys())]
        values = [str(v) for v in self.best_hyper_parameters.values()]
        hyper_parameters.append('target_metric')
        if type(self.target_metric) != str:
            values.append(self.target_metric.name)
        else:
            values.append(self.target_metric)
        data = {'Hyper Parameters': hyper_parameters, 'Values': values}
        df = pd.DataFrame(data=data)
        return df

    def get_hyper_parameters_hist_df(self):
        """
            Returns a pandas Data Frame with all hyper parameters that were selected for each model during
            the model selection process.
                :return: (Data Frame)
        """
        if not self.models_history:
            raise ModelError('Cannot get the history of hyper parameters because no model selection process '
                             'has been run. Please, run model selection first.')
        if not self.metrics:
            raise MetricError('No list of metrics was specified to train the model. '
                              'Please provide the list of metrics as a parameter of Classifier constructor.')

        # Get all validation metrics, while checking if it is for a Keras Classifier
        # or a Sci-kit learn classifier
        val_metrics = []
        for metric in self.metrics:
            if type(metric) == str:
                val_metrics.append('val_' + metric)
            else:
                val_metrics.append('val_' + metric.name)

        # Add Hyper Parameters and validation metrics that were evaluated
        hyper_parameters = [str(h) for h in list(self.hyper_parameters.keys())]
        hyper_parameters.extend(val_metrics)
        data = {'Hyper Parameters': hyper_parameters}

        for model, model_results in self.models_history.items():
            model_hyper_parameters = list(model_results['hyper_parameters'].values())
            model_metrics = [model_results['average_metric_results'][metric] for metric in val_metrics]
            model_hyper_parameters.extend(model_metrics)
            data.update([(model, model_hyper_parameters)])

        df = pd.DataFrame(data=data)
        return df

    def save_all_parameters_csv(self, save_path='parameters'):
        """"
            Saves all training, pre-processing and augmentation parameters to a CSV file.

            :param: save_path:  (str)   path to save the CSV
            :return: None.
        """
        df_manager = PandasDataFrameManager(df=self.get_hyper_parameters_df())
        df_manager.save_file(save_path=save_path + '/hyper_parameters.csv')
        df_manager = PandasDataFrameManager(df=self.get_training_parameters_df())
        df_manager.save_file(save_path=save_path + '/training_parameters.csv')
        df_manager = PandasDataFrameManager(df=self.get_pre_processing_parameters_df())
        df_manager.save_file(save_path=save_path + '/pre_processing_parameters.csv')
        df_manager = PandasDataFrameManager(df=self.get_augmentation_parameters_df())
        df_manager.save_file(save_path=save_path + '/augmentation_parameters.csv')

    def create_model(self):
        """
        Method to be implemented in children classes. (Depend on the architecture of the classifier)
            :return: None
        """

    def save_model(self, save_path):
        """
            Method that saves the classifier model in the desired location. The saving method will depend whether
            the classifier is a Keras model or a scikit-learn model. Implemented in children classes.
                :param save_path    (str)   location where the model will be saved.
                                            It must include the model name with the ".pickle" extension.
                :return: None
        """
        if not self.model:
            raise ModelError('Model has not been constructed yet! '
                             'Please build the model before saving it. Use the "create_model()" method')
        file_manager = FileManager(file=self.model)
        file_manager.save_file(save_path=save_path)

    def load_model(self, model_path):
        """
            Method that loads the classifier model from the desired location. The loading method will depend whether
            the classifier is a Keras model or a scikit-learn model. Implemented in children classes.
                :param model_path   (str)           location where the model is located.
                                                    It must include the model name with the ".pickle" extension.
                :return:            (pickle obj)    pickle object corresponding to the classifier model.
        """
        file_manager = FileManager()
        self.model = file_manager.load_file(file_path=model_path)
        return self.model


class KerasClassifier(Classifier):
    """
        Any Keras classification model (Neural Network, CNN, or RNN).
    """

    def __init__(self,
                 input_shape=None,
                 n_classes=2,
                 n_dense_layers=2,
                 dense_activation='relu',
                 n_dense_units_list=None,
                 dense_initializer='glorot_normal',
                 dense_l1=0,
                 dense_l2=0,
                 metrics=None):
        """
            Constructor of the class.
                :param input_shape:         (tuple) input tensor shape to be fed to the Neural Network.
                :param n_classes:           (int)   number of classes to be classified.
                :param n_dense_layers:      (int)   number of dense layers of the network.
                :param dense_activation:    (str)   type of neuron activation to use in the hidden dense layers.
                                                    Check the documentation of Keras for more details.
                :param n_dense_units_list:  (int)   list with length equal to the number of dense layers in the network.
                                                    Each element in the list represents the number of units per layer.
                :param dense_initializer:   (str)   Initialization method to set the dense neuron weights and biases
                                                    before training. Check the documentation of Keras for more details.
                :param dense_l1:            (float) L1 regularization constant for dense layers to avoid over fitting.
                :param dense_l2:            (float) L2 regularization constant for dense layers to avoid over fitting.
                :param metrics:             (list)  list with all Keras metrics. Each element must be a string.
                                                    Valid strings include: 'f1', 'accuracy', 'sens', 'precision'
        """

        # Avoid error if there was no metric specified
        self.target_metric = 'f1'
        if metrics is None:
            metrics = [self.target_metric]

        # For recovering the Keras metrics after loading NN instance
        # because they are deleted after saving it
        self.n_classes = n_classes
        self.metrics_str = metrics
        metrics = self.get_keras_metrics(metrics)
        super().__init__(n_classes=n_classes,
                         metrics=metrics,
                         input_shape=input_shape)

        # Guarantee a list of units is passed
        if n_dense_units_list is None:
            n_dense_units_list = [np.random.randint(50, 100) for _ in range(n_dense_layers)]
        elif type(n_dense_units_list) is not list:
            raise ValueError('Specify a list of number of neurons per layer. '
                             'Instead type {} was passed!'.format(type(n_dense_units_list)))

        # Guarantee that the units match the number of layers
        if len(n_dense_units_list) is not n_dense_layers:
            raise ModelError('The length of the list of units ({}) '
                             'does not match the number of hidden layers ({})'.format(len(n_dense_units_list),
                                                                                      n_dense_layers))

        if n_dense_layers < 2:
            raise ModelError('Cannot create a network with {} layers. '
                             'You must specify a number greater than two, which includes both, '
                             'the input and hidden layers.')

        self.type_classifier = 'keras'
        self.model_history = {}
        self.target_metric = custom_km.F1(name='f1')
        self.input_shape = input_shape

        # Set dense topology
        self.n_dense_layers = n_dense_layers
        self.n_dense_units_list = n_dense_units_list
        self.dense_activation = dense_activation
        self.dense_regularizer = regularizers.l1_l2(l1=dense_l1, l2=dense_l2)
        self.dense_initializer = dense_initializer

        # Set optimization parameters
        self.lr = 0.001
        self.batch_size = 32
        self.epochs = 10
        self.k_folds = 1
        self.optimizer = optimizers.get('adam')
        self.optimizer.learning_rate = self.lr

        # Create hyper parameter dictionary
        self.hyper_parameters['lr'] = self.lr
        self.hyper_parameters['batch_size'] = self.batch_size
        self.hyper_parameters['epochs'] = self.epochs
        self.hyper_parameters['k_folds'] = self.k_folds
        self.hyper_parameters['optimizer'] = 'adam'
        self.hyper_parameters['n_dense_layers'] = n_dense_layers
        self.hyper_parameters['dense_units'] = str(n_dense_units_list)
        self.hyper_parameters['dense_l1'] = dense_l1
        self.hyper_parameters['dense_l2'] = dense_l2
        self.hyper_parameters['dense_activation'] = dense_activation
        self.hyper_parameters['dense_initializer'] = dense_initializer

        # Initialize best hyper parameters
        self.best_hyper_parameters['lr'] = self.lr
        self.best_hyper_parameters['batch_size'] = self.batch_size
        self.best_hyper_parameters['epochs'] = self.epochs
        self.best_hyper_parameters['k_folds'] = self.k_folds
        self.best_hyper_parameters['optimizer'] = 'adam'
        self.best_hyper_parameters['model'] = 0
        self.best_hyper_parameters['n_dense_layers'] = n_dense_layers
        self.best_hyper_parameters['dense_units'] = str(n_dense_units_list)
        self.best_hyper_parameters['dense_l1'] = dense_l1
        self.best_hyper_parameters['dense_l2'] = dense_l2
        self.best_hyper_parameters['dense_activation'] = dense_activation
        self.best_hyper_parameters['dense_initializer'] = dense_initializer

    def save_model(self, save_path, save_best=False):
        """
            Method that saves the classifier model in the desired location.
                :param save_path    (str)   location where the model will be saved.
                                            It must include the model name without extension.
                :param save_best:   (bool)  whether to save the best model or the last trained model.
                :return: None
        """

        model = self.model
        if save_best:
            model = self.best_model

        if not model:
            raise ModelError('Model has not been constructed yet! '
                             'Please build the model before saving it. Use the "create_model()" method')
        if '.' in save_path:
            raise ValueError('The save path for the Keras model cannot contain any extension! '
                             'Remove the extension before saving the model.')

        # Create the respective saving paths
        architecture_path = save_path + '.json'
        weights_path = save_path + '.h5'

        # Save the trained model architecture
        json_model = model.to_json()
        json_file_manager = JsonFileManager(file=json_model)
        json_file_manager.save_file(save_path=architecture_path)

        # Save the model weights
        model.save_weights(weights_path)

        # Save model architecture schematic
        with open(save_path + '_architecture.txt', 'w+') as f:
            with redirect_stdout(f):
                model.summary()

    def load_model(self, model_path, load_best=False):
        """
            Method that loads the classifier model from the desired location.
                :param model_path   (str)   location where the model resides.
                                            It must include the model name without extension.
                :param load_best:   (bool)  whether to set the loaded model as the best model.

                :return: model      (Keras model)   Keras model.
        """
        if '.' in model_path:
            raise ValueError('The model path for the Keras model cannot contain any extension! '
                             'Remove the extension before loading the model.')

        # Load model architecture
        json_file_manager = JsonFileManager()
        json_model = json_file_manager.load_file(file_path=model_path + '.json')
        metrics_dict = {m.name: m for m in self.get_keras_metrics(metrics=self.metrics_str) if type(m) != str}
        self.model = model_from_json(json_model, custom_objects=metrics_dict)

        # Load weights into the model
        self.model.load_weights(model_path + '.h5')

        if load_best:
            self.best_model = self.model
        return self.model

    def reset_keras_models(self):
        """
            Method that cleans all keras models, to allow saving the neural network instance.
            Be sure that all keras models are saved before calling this method.
                :return: None
        """
        warnings.warn('WARNING! All trained Keras models are going to be deleted!')
        self.model = None
        self.trained_models = None
        self.best_model = None
        self.metrics = self.metrics_str

    def get_keras_metrics(self, metrics):
        """
            Method that configures the list of metrics in the KerasClassifier.
                :param metrics: (list)  list with all metrics. Each element must be a string. Valid strings include:
                                        'f1', 'sensitivity', 'specificity', 'accuracy', 'precision', 'auc', 'loss'
                :return:        (list)  list with all Keras metrics. Each element is a custom Keras metric object.
                                        For more details check: https://keras.io/api/metrics/
        """

        # Set metrics
        if metrics is None:
            keras_metrics = ['accuracy']
        else:
            # Initialize list
            keras_metrics = []

            # Check that metrics are well specified
            valid_metrics = ['f1', 'sensitivity', 'specificity', 'accuracy', 'precision', 'auc', 'loss']

            # Check for valid metrics if the classifier is not binary
            binary_metrics_only = ['f1', 'sens', 'precision', 'auc']
            invalid_metrics = [metric for metric in metrics if metric in binary_metrics_only]
            if self.n_classes > 2 and invalid_metrics:
                raise MetricError('MetricError: Cannot implement the metrics: {} '
                                  'for a classifier that is not binary.'.format(invalid_metrics))

            for metric in metrics:
                if metric.lower() not in valid_metrics:
                    raise ValueError("ValueError: invalid metric in list of specified metrics: '{}'.\n"
                                     "Valid metrics include: {}".format(metric, valid_metrics))
                # Assign metrics
                if metric == 'auc':
                    keras_metrics.append(km.AUC(name=metric))
                elif metric == 'accuracy':
                    keras_metrics.append('accuracy')
                elif metric == 'sensitivity':
                    keras_metrics.append(km.Recall(name=metric))
                elif metric == 'specificity':
                    keras_metrics.append(custom_km.Specificity(name=metric))
                elif metric == 'precision':
                    keras_metrics.append(km.Precision(name=metric))
                elif metric == 'f1':
                    keras_metrics.append(custom_km.F1(name='f1'))
                elif metric == 'loss':
                    keras_metrics.append(custom_km.F1(name='loss'))

        return keras_metrics

    def fit_generator(self,
                      generator=generators.ClassifierImageGenerator([], {}),
                      batch_size=32,
                      epochs=1,
                      k_folds=1,
                      lr=0.001,
                      optimizer_name='adam',
                      callbacks=None,
                      validation_split=0.7,
                      shuffle=True,
                      save_path=None,
                      verbose=2):
        """
            Trains by using generators.

                :param generator:           (Generator) generator Object with all training and validation data file
                                                        names. Current valid generators are: ClassifierImageGenerator
                :param batch_size:          (int)       size of the batch to generate at each iteration of the data
                                                        generation.
                :param epochs:              (int)       number of iterations that the neural network will be trained
                                                        with all the data
                :param k_folds:             (int)       number of "k" random data divisions to train the model "k" times
                :param lr:                  (float)     learning constant to update gradient steps during training.
                :param optimizer_name:      (str)       name of the optimizer. Valid names include: 'Adam', 'SGD', 'RMSprop'
                                                        Check the documentation of Keras for more details.
                :param callbacks:           (list)      list of Keras.Callbacks. Ckeck Keras documentation for details:
                                                        https://keras.io/api/callbacks/
                :param validation_split:    (float)     fraction of data to be defined for training.
                                                        If k_folds is larger than one, the argument will be ignored.
                :param shuffle:             (bool)      flag to indicate if the data will be randomized.
                :param save_path:           (str)       folder path where the results will be saved.
                :param verbose:             (int)       how much information will be displayed when training.
                                                        [Valid arguments: '0', '1' or '2']
                :return None
        """

        # Get all data from generator
        x = generator.x_file_paths
        y = generator.labels_dict

        if type(x) != list:
            raise ValueError('"x_file_paths" data must be a list with all file paths of training samples!')

        if type(y) != dict:
            raise ValueError('"labels_dict" data must be a dictionary with keys as the file paths of the '
                             'training samples!')

        #  Setting pre-processing attributes
        self.pre_processor = generator.pre_processor

        #  Setting augmentation attributes
        self.transformer = generator.transformer

        # Initialize two generators for training and validation
        training_generator = generator
        validation_generator = deepcopy(training_generator)

        # Release resources
        del generator
        gc.collect()

        # Only augment training data
        validation_generator.transformer = None

        # Assign training parameters
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_folds = k_folds
        self.type_generator = type(training_generator)

        # Assign optimization parameters
        self.optimizer = optimizers.get(optimizer_name)
        self.optimizer.learning_rate = self.lr

        # Update dictionary of hyper parameters
        self.hyper_parameters['lr'] = self.lr
        self.hyper_parameters['batch_size'] = self.batch_size
        self.hyper_parameters['epochs'] = self.epochs
        self.hyper_parameters['k_folds'] = self.k_folds
        self.hyper_parameters['optimizer'] = optimizer_name

        # Update verbosity
        self.verbose = verbose

        # Generate random k splits
        if k_folds == 1:
            # Get all indices of training samples
            data_idx = list(range(len(x)))

            # Shuffle the indices
            random.shuffle(data_idx)

            # Get new randomized data
            x = np.array(x)[data_idx]
            x = list(x)
            y = {file_path: y[file_path] for file_path in x}

            # Split the data
            middle_idx = int(round(validation_split * len(x)))
            train_idx = data_idx[0:middle_idx]
            test_idx = data_idx[middle_idx::]

            # Get the folds
            folds = [(train_idx, test_idx)]
        else:
            y_list = [y[file_path] for file_path in x]
            folds = list(StratifiedKFold(n_splits=k_folds, shuffle=True).split(x, y_list))

        # Save general training, pre-processing and augmentation parameters
        if save_path:
            self.save_all_parameters_csv(save_path=save_path)

        for i, (train_idx, test_idx) in enumerate(folds):

            # Print message for model selection
            if verbose == 0 or verbose == 2:
                print('\n-------------------------------------------------------')
                print('FOLD {}'.format(i + 1))
                print('-------------------------------------------------------')

            # Get data for specific k fold
            x_train = np.array(x)[train_idx]
            x_train = list(x_train)
            y_train = {file_path: y[file_path] for file_path in x_train}

            x_test = np.array(x)[test_idx]
            x_test = list(x_test)
            y_test = {file_path: y[file_path] for file_path in x_test}

            # Create model
            lock = threading.Lock()
            lock.acquire()
            K.clear_session()
            self.create_model()
            # Compile model
            self.model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics)
            lock.release()

            if verbose == 0 or verbose == 2:
                print('Training with {} samples. Validating with {} samples.'.format(len(x_train),
                                                                                     len(x_test)))

            # Initialize Training Generator with k-fold data
            training_generator.x_file_paths = x_train
            training_generator.labels_dict = y_train
            training_generator.indexes = np.arange(len(x_train))
            training_generator.num_batches = 0
            training_generator.current_batch = 0
            training_generator.n_epoch = 1
            training_generator.on_epoch_end()

            # Initialize Validation Generator with k-fold data
            validation_generator.x_file_paths = x_test
            validation_generator.labels_dict = y_test
            validation_generator.indexes = np.arange(len(x_test))
            validation_generator.num_batches = 0
            validation_generator.current_batch = 0
            validation_generator.n_epoch = 1
            validation_generator.on_epoch_end()

            # Train with generators
            history = self.model.fit(x=training_generator,
                                     validation_data=validation_generator,
                                     epochs=epochs,
                                     verbose=verbose,
                                     callbacks=callbacks,
                                     shuffle=shuffle,
                                     workers=os.cpu_count())

            # Print message for model selection
            if verbose == 0 or verbose == 2:
                print('-------------------------------------------------------')
                print('Validation loss fold {}: {}'.format(i + 1, history.history['val_loss'][-1]))

            self.model_history['fold_{}'.format(i)] = history.history

            if save_path:
                print('-------------------------------------------------------')
                print('Saving Fold {} Results'.format(i + 1))
                df_manager = PandasDataFrameManager(df=self.get_model_hist_df())
                df_manager.save_file(save_path=save_path + '/model_history_results.csv')
                df_manager = PandasDataFrameManager(df=self.get_average_metrics_hist_df())
                df_manager.save_file(save_path=save_path + '/average_history_results.csv')
                self.save_model(save_path=save_path + '/model')

        if save_path:
            print('\n-------------------------------------------------------')
            print('SAVING MODEL RESULTS')
            print('-------------------------------------------------------')
            df_manager = PandasDataFrameManager(df=self.get_model_hist_df())
            df_manager.save_file(save_path=save_path + '/model_history_results.csv')
            df_manager = PandasDataFrameManager(df=self.get_average_metrics_hist_df())
            df_manager.save_file(save_path=save_path + '/average_history_results.csv')
            self.save_model(save_path=save_path + '/model')

    def fit(self,
            x,
            y,
            batch_size=32,
            epochs=1,
            k_folds=1,
            lr=0.001,
            optimizer_name='adam',
            callbacks=None,
            validation_split=0.7,
            shuffle=True,
            pre_processor=pre_processors.PreProcessor(),
            transformer=transformers.TransformerClassifier(),
            save_path=None,
            verbose=2):

        """
            Method that trains the neural network.
                :param x:                   (np_array)      input data with shape equal to the attribute of the
                                                            class self.input_shape.
                :param y:                   (np_array)      output data with shape [n_samples,]
                :param batch_size:          (int)           number of data samples to train within each epoch.
                :param epochs:              (int)           number of iterations that the neural network will be trained
                                                            with all the data
                :param k_folds:             (int)           number of "k" random data divisions to train the model "k"
                                                            times.
                :param lr:                  (float)         learning constant to update gradient steps during training.
                :param optimizer_name:      (str)           name of the optimizer. Valid names include: 'Adam', 'SGD', 'RMSprop'
                                                            Check the documentation of Keras for more details.
                :param callbacks:           (list)          list of Keras.Callbacks. Ckeck Keras documentation for
                                                            details: https://keras.io/api/callbacks/
                :param validation_split:    (float)         fraction of data to be defined for training.
                                                            If k_folds is larger than one, the argument will be ignored.
                :param shuffle:             (bool)          flag to indicate if the data will be randomized.
                :param pre_processor        (PreProcessor)  pre-processor object that will pre-process the data.
                                                            Check the documentation in the utils module.
                :param transformer:         (Transformer)   transformer object that will augment the data. Check the
                                                            documentation in the transformers module.
                :param save_path:           (str)           folder path where the results will be saved.
                :param verbose:             (int)           how much information will be displayed when training.
                                                            [Valid arguments: '0', '1' or '2']
                :return: None
        """

        if type(x) != np.ndarray or type(y) != np.ndarray:
            raise ValueError('"x" and "y" data must be numpy arrays!')

        if len(y.shape) != 1:
            raise DimensionalityError('Invalid shape for "y" data numpy array! '
                                      'Data needs to have the following shape: [n_samples,]')

        # Assign training parameters
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_folds = k_folds

        # Assign optimization parameters
        self.optimizer = optimizers.get(optimizer_name)
        self.optimizer.learning_rate = self.lr

        # Update dictionary of hyper parameters
        self.hyper_parameters['lr'] = self.lr
        self.hyper_parameters['batch_size'] = self.batch_size
        self.hyper_parameters['epochs'] = self.epochs
        self.hyper_parameters['k_folds'] = self.k_folds
        self.hyper_parameters['optimizer'] = optimizer_name

        # Update verbosity = x
        self.verbose = verbose

        #  Setting pre-processing attributes
        self.pre_processor = pre_processor

        #  Setting augmentation attributes
        self.transformer = transformer

        # Generate random k splits
        if k_folds == 1:
            # Get all indices of training samples
            data_idx = list(range(len(x)))

            # Shuffle the indices
            random.shuffle(data_idx)

            # Get new randomized data
            x = x[data_idx]
            y = y[data_idx]

            # Split the data
            middle_idx = int(round(validation_split * len(x)))
            train_idx = data_idx[0:middle_idx]
            test_idx = data_idx[middle_idx::]

            # Get the folds
            folds = [(train_idx, test_idx)]
        else:
            folds = list(StratifiedKFold(n_splits=k_folds, shuffle=True).split(x, y))

        # Save general training, pre-processing and augmentation parameters
        if save_path:
            self.save_all_parameters_csv(save_path=save_path)

        for i, (train_idx, test_idx) in enumerate(folds):

            # Print message for model selection
            if verbose == 0 or verbose == 2:
                print('\n-------------------------------------------------------')
                print('FOLD {}'.format(i + 1))
                print('-------------------------------------------------------')

            # Get data for specific k fold
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_test = x[test_idx]
            y_test = y[test_idx]

            if verbose == 0 or verbose == 2:
                print('Augmenting data with shape: {}.'.format(x_train.shape))
            x_train, y_train = self.transformer.augment_data(x_train, y_train)

            # Pre-process data
            if verbose == 0 or verbose == 2:
                print('Pre-processing training data with shape: {}.'.format(x_train.shape))
                print('Pre-processing test data with shape: {}.'.format(x_test.shape))
            x_train = pre_processor.pre_process(x_train)
            x_test = pre_processor.pre_process(x_test)

            # Create model
            K.clear_session()
            self.create_model()

            # Compile model
            self.model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics)

            # Ensure valid format is passed to fit method
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            x_test = np.asarray(x_test)
            y_test = np.asarray(y_test)

            if verbose == 0 or verbose == 2:
                print('Training with {} samples. Validating with {} samples.'.format(x_train.shape[0],
                                                                                     x_test.shape[0]))

            history = self.model.fit(x=x_train,
                                     y=y_train,
                                     validation_data=(x_test, y_test),
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     verbose=verbose,
                                     callbacks=callbacks,
                                     shuffle=shuffle)

            # Print message for model selection
            if verbose == 0 or verbose == 2:
                print('-------------------------------------------------------')
                print('Validation loss fold {}: {}'.format(i + 1, history.history['val_loss'][-1]))

            self.model_history['fold_{}'.format(i)] = history.history

            if save_path:
                print('-------------------------------------------------------')
                print('Saving Fold {} Results'.format(i + 1))
                df_manager = PandasDataFrameManager(df=self.get_model_hist_df())
                df_manager.save_file(save_path=save_path + '/model_history_results.csv')
                df_manager = PandasDataFrameManager(df=self.get_average_metrics_hist_df())
                df_manager.save_file(save_path=save_path + '/average_history_results.csv')
                self.save_model(save_path=save_path + '/model')

        if save_path:
            print('\n-------------------------------------------------------')
            print('SAVING MODEL RESULTS')
            print('-------------------------------------------------------')
            df_manager = PandasDataFrameManager(df=self.get_model_hist_df())
            df_manager.save_file(save_path=save_path + '/model_history_results.csv')
            df_manager = PandasDataFrameManager(df=self.get_average_metrics_hist_df())
            df_manager.save_file(save_path=save_path + '/average_history_results.csv')
            self.save_model(save_path=save_path + '/model')

    def model_selection(self,
                        x,
                        y,
                        target_metric='f1',
                        search_mode='random',
                        n_models=5,
                        hyper_parameter_space=None,
                        epochs=50,
                        k_folds=10,
                        callbacks=None,
                        validation_split=0.7,
                        shuffle=True,
                        pre_processor=pre_processors.PreProcessor(),
                        transformer=transformers.TransformerClassifier(),
                        save_path=None,
                        verbose=2):
        """
            :param x:                       (array) input data with shape equal to [n_samples, n_fetaures]

                                            (Generator) generator Object with all training and validation data file
                                                        names. Current valid generators are: ClassifierImageGenerator

            :param y:                       (array) output data with shape [n_samples,]
            :param target_metric:           (str)   metric to consider when selecting the best model. Valid strings are:
                                                    'f1', 'sensitivity', 'specificity', 'accuracy', 'precision', 'auc'
            :param search_mode:             (str)   whether the search is random or bayesian.
                                                    Valid strings include: 'random', 'bayesian'
            :param n_models:                (int)   number of models to search.
            :param epochs                   (int)   Number of epochs to train the model.
                                                    An epoch is an iteration over the entire x and y data provided.
            :param k_folds:                 (int)   number of cross-validation splits to make when training each model.
                                                    This is used when there is few training data available.
            :param hyper_parameter_space:   (dict)  dictionary where each key represent the hyper parameter to search.
                                                    The dictionary values can represent either a list, a tuple, or an int
                                                    depending on the hyper parameter.
                                                    Valid dictionary keys/value pairs include:

                                                    'lr':                   (int)           range: [-infinity, 0]
                                                    'batch_size':           (list of int)   range: [1, infinity]
                                                    'n_layers':             (list of int)   range: [2, infinity]
                                                    'n_units':              (list of int)   range: [2, infinity]
                                                    'hidden_activation':    (list of str)   ['relu', 'elu', 'tanh', ...]
                                                    'optimizer':            (list of str)   ['Adam', 'SGD', 'RMSprop,...]
                                                    'initializer':          (list of str)   ['glorot_normal', ...]
                                                    'regularizer':          (int)           range: [-infinity, 0]

                                                    (Example: {'n_layers': (10, 100)} will search for the best
                                                    number of layers between 10 layers and 100 layers).
            :param callbacks:               (list)  list of Keras.Callbacks. Ckeck Keras documentation for details:
                                                    https://keras.io/api/callbacks/
            :param validation_split:        (float) fraction of data to be defined for training. If k_folds is larger
                                                    than one, the argument will be ignored.
            :param shuffle:                 (bool)  flag to indicate if the data will be randomized.
            :param pre_processor            (PreProcessor)  pre-processor object that will pre-process the data.
                                                            Check the documentation in the utils module.
            :param transformer:             (Transformer)   transformer object that will augment the data. Check the
                                                            documentation in the transformers module.
            :param save_path:               (str)   folder path where the results will be saved.
            :param verbose:                 (int)   how much information will be displayed when training.
                                                    [Valid arguments: '0', '1' or '2']
            :return: None
        """

        # Get list of hyper parameters
        valid_hyper_parameters = ['lr',
                                  'batch_size',
                                  'n_dense_layers',
                                  'dense_units',
                                  'dense_activation',
                                  'optimizer',
                                  'dense_initializer',
                                  'dense_regularizer']
        hyper_parameter_names = list(hyper_parameter_space.keys())

        # Check for invalid hyper parameter names
        invalid_names = set(valid_hyper_parameters).union(set(hyper_parameter_names)) - \
                        set(valid_hyper_parameters).intersection(set(hyper_parameter_names)) - \
                        set(valid_hyper_parameters)

        if invalid_names:
            raise KeyError("You entered the following invalid hyper-parameter names: {}. "
                           "Valid hyper-parameters include: {}".format(str(invalid_names), str(valid_hyper_parameters)))

        # Check if metric is valid
        valid_metrics = ['f1', 'sensitivity', 'specificity', 'accuracy', 'precision', 'auc', 'loss']
        if target_metric not in valid_metrics:
            raise ValueError("Invalid metric name '{0}'. "
                             "Valid metric names include: {1}".format(target_metric, valid_metrics))

        # Check if search mode is valid
        valid_search = ['random', 'bayesian']
        if search_mode not in valid_search:
            raise ValueError("ValueError: invalid search mode '{0}'. "
                             "Valid search modes include: {1}".format(search_mode, valid_search))

        # Check if target metric is valid
        target_metric = target_metric.lower()
        valid_metrics = ['f1', 'sensitivity', 'specificity', 'accuracy', 'precision', 'auc', 'loss']
        if target_metric not in valid_metrics:
            raise ValueError("ValueError: invalid target metric '{0}'.\n"
                             "Valid metrics include: {1}".format(target_metric, valid_metrics))

        # Check for valid metrics for the target if not the classifier is not binary
        binary_metrics = ['f1', 'sensitivity', 'specificity', 'precision', 'auc']
        if self.n_classes > 2 and target_metric in binary_metrics:
            raise MetricError('MetricError: Cannot specify the target metric: {} '
                              'for a binary classifier.'.format(target_metric))

        # Initialize hyper parameter spaces
        # Avoid conflicts, if the user did not specified any of the following hyper parameters
        lr_space = -5
        batch_size_space = [16, 32, 64, 128]
        n_dense_layers_space = list(range(3, 5, 1))
        dense_units_space = list(range(20, 200, 20))
        dense_regularizer_space = -5
        dense_activation_space = ['relu', 'elu', 'tanh']
        dense_initializer_space = ['glorot_normal', 'glorot_uniform']
        optimizer_space = ['Adam', 'SGD', 'RMSprop']

        # Get search space for hyper parameters
        for hyper_parameter, search_space in hyper_parameter_space.items():
            if hyper_parameter == 'lr':
                lr_space = search_space
            elif hyper_parameter == 'batch_size':
                batch_size_space = search_space
            elif hyper_parameter == 'n_dense_layers':
                n_dense_layers_space = search_space
            elif hyper_parameter == 'dense_units':
                dense_units_space = search_space
            elif hyper_parameter == 'dense_regularizer':
                dense_regularizer_space = search_space
            elif hyper_parameter == 'dense_activation':
                dense_activation_space = search_space
            elif hyper_parameter == 'dense_initializer':
                dense_initializer_space = search_space
            elif hyper_parameter == 'optimizer':
                optimizer_space = search_space

        # Get final hyper parameter space dictionary
        self.hyper_parameter_space['lr'] = lr_space
        self.hyper_parameter_space['batch_size'] = batch_size_space
        self.hyper_parameter_space['optimizer'] = optimizer_space
        self.hyper_parameter_space['n_dense_layers'] = n_dense_layers_space
        self.hyper_parameter_space['dense_units'] = dense_units_space
        self.hyper_parameter_space['dense_regularizer'] = dense_regularizer_space
        self.hyper_parameter_space['dense_activation'] = dense_activation_space
        self.hyper_parameter_space['dense_initializer'] = dense_initializer_space

        # Assign metrics
        if target_metric == 'auc':
            self.target_metric = km.AUC(name=target_metric)
        elif target_metric == 'accuracy':
            self.target_metric = 'accuracy'
        elif target_metric == 'sensitivity':
            self.target_metric = km.Recall(name=target_metric)
        elif target_metric == 'specificity':
            self.target_metric = custom_km.Specificity(name=target_metric)
        elif target_metric == 'precision':
            self.target_metric = km.Precision(name=target_metric)
        elif target_metric == 'f1':
            self.target_metric = custom_km.F1(name=target_metric)
        elif target_metric == 'loss':
            self.target_metric = 'loss'

        # Initialize best metric score
        if target_metric == 'loss':
            best_metric_score = np.inf
        else:
            best_metric_score = 0

        # Search for best model
        for i in range(n_models):

            # Initialize model dictionary to save history
            self.models_history['model_{}'.format(i)] = {}

            # Sample hyper parameters from search spaces
            lr = (10 ** (lr_space * (np.random.rand()))) / 10
            batch_size = random.choice(batch_size_space)
            optimizer = random.choice(optimizer_space)
            n_dense_layers = random.choice(n_dense_layers_space)
            dense_units = [random.choice(dense_units_space) for _ in range(n_dense_layers)]
            dense_l1 = (10 ** (dense_regularizer_space * (np.random.rand()))) / 10
            dense_l2 = (10 ** (dense_regularizer_space * (np.random.rand()))) / 10
            dense_activation = random.choice(dense_activation_space)
            dense_initializer = random.choice(dense_initializer_space)

            # Save hyper parameter configuration in history
            self.models_history['model_{}'.format(i)]['hyper_parameters'] = {}
            self.models_history['model_{}'.format(i)]['hyper_parameters']['lr'] = lr
            self.models_history['model_{}'.format(i)]['hyper_parameters']['batch_size'] = batch_size
            self.models_history['model_{}'.format(i)]['hyper_parameters']['epochs'] = epochs
            self.models_history['model_{}'.format(i)]['hyper_parameters']['k_folds'] = k_folds
            self.models_history['model_{}'.format(i)]['hyper_parameters']['optimizer'] = optimizer
            self.models_history['model_{}'.format(i)]['hyper_parameters']['n_dense_layers'] = n_dense_layers
            self.models_history['model_{}'.format(i)]['hyper_parameters']['dense_units'] = str(dense_units)
            self.models_history['model_{}'.format(i)]['hyper_parameters']['dense_l1'] = dense_l1
            self.models_history['model_{}'.format(i)]['hyper_parameters']['dense_l2'] = dense_l2
            self.models_history['model_{}'.format(i)]['hyper_parameters']['dense_activation'] = dense_activation
            self.models_history['model_{}'.format(i)]['hyper_parameters']['dense_initializer'] = dense_initializer

            # Save last hyper parameters
            self.hyper_parameters['lr'] = lr
            self.hyper_parameters['batch_size'] = batch_size
            self.hyper_parameters['epochs'] = epochs
            self.hyper_parameters['k_folds'] = k_folds
            self.hyper_parameters['optimizer'] = optimizer
            self.hyper_parameters['n_dense_layers'] = n_dense_layers
            self.hyper_parameters['dense_units'] = str(dense_units)
            self.hyper_parameters['dense_l1'] = dense_l1
            self.hyper_parameters['dense_l2'] = dense_l2
            self.hyper_parameters['dense_activation'] = dense_activation
            self.hyper_parameters['dense_initializer'] = dense_initializer

            # Set topology
            self.input_shape = self.input_shape
            self.n_dense_layers = n_dense_layers
            self.n_dense_units_list = dense_units
            self.dense_activation = dense_activation

            # Set optimization parameters
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.optimizer = optimizers.get(optimizer)
            self.optimizer.learning_rate = lr
            self.dense_regularizer = regularizers.l1_l2(l1=dense_l1, l2=dense_l2)
            self.dense_initializer = dense_initializer

            # Train
            print('\n\t--------------------------------------------------------')
            print('\t\t\t\t\t\tTRAINING MODEL {}'.format(i))
            print('\t--------------------------------------------------------\n')

            # Detect if a generator will be used instead of classic training
            if type(x) == generators.ClassifierImageGenerator:
                self.fit_generator(generator=x,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   k_folds=k_folds,
                                   validation_split=validation_split,
                                   shuffle=shuffle,
                                   save_path=save_path + '/model_{}'.format(i),
                                   callbacks=callbacks,
                                   verbose=verbose)
            else:
                self.fit(x,
                         y,
                         batch_size=batch_size,
                         epochs=self.epochs,
                         k_folds=k_folds,
                         validation_split=validation_split,
                         shuffle=shuffle,
                         pre_processor=pre_processor,
                         transformer=transformer,
                         save_path=save_path + '/model_{}'.format(i),
                         callbacks=callbacks,
                         verbose=verbose)

            # Compute results of model
            self.models_history['model_{}'.format(i)]['history'] = self.model_history
            self.models_history['model_{}'.format(i)]['average_metric_history'] = self.get_average_metric_hist()
            self.models_history['model_{}'.format(i)]['average_metric_results'] = self.get_average_metric_results()
            metric_score = self.models_history['model_{}'.format(i)]['average_metric_results']['val_' + target_metric]

            # Update list of trained models
            self.trained_models.append(self.model)

            # Update best model configuration
            if metric_score <= best_metric_score and target_metric == 'loss':
                best_metric_score = metric_score
                self.best_model = self.model
                self.best_hyper_parameters['model'] = i
                self.best_hyper_parameters['lr'] = lr
                self.best_hyper_parameters['batch_size'] = batch_size
                self.best_hyper_parameters['epochs'] = epochs
                self.best_hyper_parameters['k_folds'] = k_folds
                self.best_hyper_parameters['optimizer'] = optimizer
                self.best_hyper_parameters['n_dense_layers'] = n_dense_layers
                self.best_hyper_parameters['dense_units'] = str(dense_units)
                self.best_hyper_parameters['dense_l1'] = dense_l1
                self.best_hyper_parameters['dense_l2'] = dense_l2
                self.best_hyper_parameters['dense_activation'] = dense_activation
                self.best_hyper_parameters['dense_initializer'] = dense_initializer

            elif metric_score > best_metric_score and target_metric != 'loss':
                best_metric_score = metric_score
                self.best_model = self.model
                self.best_hyper_parameters['model'] = i
                self.best_hyper_parameters['lr'] = lr
                self.best_hyper_parameters['batch_size'] = batch_size
                self.best_hyper_parameters['epochs'] = epochs
                self.best_hyper_parameters['k_folds'] = k_folds
                self.best_hyper_parameters['optimizer'] = optimizer
                self.best_hyper_parameters['n_dense_layers'] = n_dense_layers
                self.best_hyper_parameters['dense_units'] = str(dense_units)
                self.best_hyper_parameters['dense_l1'] = dense_l1
                self.best_hyper_parameters['dense_l2'] = dense_l2
                self.best_hyper_parameters['dense_activation'] = dense_activation
                self.best_hyper_parameters['dense_initializer'] = dense_initializer

            if save_path:
                df_manager = PandasDataFrameManager(df=self.get_hyper_parameters_hist_df())
                df_manager.save_file(save_path=save_path + '/hyper_parameter_history.csv')
                df_manager = PandasDataFrameManager(df=self.get_best_hyper_parameters_df())
                df_manager.save_file(save_path=save_path + '/best_hyper_parameters.csv')
                df_manager = PandasDataFrameManager(df=self.get_dict_metric_comparison())
                df_manager.save_multiple_df(save_path=save_path + '/models_average_history_comparison.xlsx')
                if self.best_model is not None:
                    self.save_model(save_path=save_path + '/best_model', save_best=True)

            print('\n-------------------------------------------------------')
            print('CURRENT {} SCORE: {}'.format(target_metric.upper(), metric_score))
            print('BEST {} SCORE: {}'.format(target_metric.upper(), best_metric_score))
            print('-------------------------------------------------------')

        # Make the last model be the best model
        self.model = self.best_model

        # TODO: implement bayesian search
        if 'bayesian' == search_mode:
            raise NotImplementedError('Bayesian Optimization has not been implemnted yet')

    def predict_proba(self, x, use_best_model=False):

        """
            Method that predicts the labels based on the corresponding matrix of data 'x'.
                :param x:               [np_array]  matrix of training data
                :param use_best_model:  [bool]      whether to use the best model to make predictions
                                                    or use the last trained model
                :return y:              [np_array]  predicted labels.
        """

        if type(x) != np.ndarray:
            raise ValueError('"x" data must be numpy array!')

        model = self.model
        if use_best_model:
            model = self.best_model

        if model is None:
            raise ModelError('Cannot predict samples because no model has been constructed! '
                             'Please build a model before predicting')

        x = self.pre_processor.pre_process(x)

        # Convert Grayscale to RGB
        if len(self.x_original_shape) == 4 and len(x.shape) == 3:
            x = np.array([np.array(Image.fromarray(img).convert('RGB')) for img in x])

        # Check that the  input training data has the same shape
        # as the data that was trained before applying pre-processing techniques
        if x.shape[1::] != self.x_original_shape[1::]:
            raise DimensionalityError('Invalid shape of {} for pre_processed "x" numpy array! '
                                      'Data needs to have the following shape: {}'.format(x.shape[1::],
                                                                                          self.x_original_shape[1::]))

        return model.predict(x)

    def get_average_metric_hist(self):
        """
            Returns a dictionary with the average training history  of all model metrics.
            The keys correspond to the metric names and the values to the list of metric scores over all epochs.

                :return: Dictionary with metrics as key values and the average list of scores as the corresponding value
        """
        if not self.model_history:
            raise ModelError('Cannot collect metrics from a model that has not been trained!'
                             ' Run the "fit()" method before collecting the metrics')

        # Get all folds and all metrics evaluated
        folds = list(self.model_history.keys())
        metric_names = list(self.model_history[folds[0]])

        # Initialize dictionary of metrics and of average results
        metric_dict = {metric: [] for metric in metric_names}
        avg_metric_hist_dict = {metric: 0 for metric in metric_names}

        # Append list of all values of metrics in the training history of each fold
        for fold in folds:
            for metric in metric_names:
                metric_dict[metric].append(self.model_history[fold][metric])

        for metric in metric_names:
            avg_metric_hist_dict[metric] = list(np.average(metric_dict[metric], axis=0))

        return avg_metric_hist_dict

    def get_average_metric_results(self):

        """
            Returns a dictionary with the final average of all model metrics obtained at the last epoch.
            The average is computed with the result obtained at the last epoch of each k fold training iteration.

                :return: Dictionary with metrics as key values and the average result as the corresponding value.
        """

        # Get average metrics in all epochs
        avg_metric_hist_dict = self.get_average_metric_hist()
        metric_names = list(avg_metric_hist_dict.keys())
        avg_dict = {}
        for metric in metric_names:
            avg_dict[metric] = avg_metric_hist_dict[metric][-1]
        return avg_dict

    def get_model_hist_df(self):
        """
            Returns the results of training and validation history of the model for all k folds.
            The data is condensed in a pandas DataFrame.
                :return: df_model_results   (Data Frame)
        """

        if not self.model_history:
            raise ModelError('Cannot return the pandas Data Frame with the model results'
                             'because no training has happened! Train the model first!')

        # Initialize
        folds = list(self.model_history.keys())
        metrics = list(self.model_history[folds[0]].keys())
        indices = ['Epoch {}'.format(i + 1) for i in range(len(self.model_history[folds[0]][metrics[0]]))]
        df_model_results = pd.DataFrame(index=indices)

        # Create Data Frame
        for i, f in enumerate(folds):
            for m in metrics:
                column = 'Fold {} - {}'.format(i, m)
                df_model_results[column] = self.model_history[f][m]
        return df_model_results

    def get_average_metrics_hist_df(self):
        """
            Returns the average metrics results of the model (computed over all k folds).
            The data is condensed in a pandas DataFrame.
                :return: df_avg_metrics   (Data Frame)
        """
        avg_metrics_dict = self.get_average_metric_hist()
        df_avg_metrics = pd.DataFrame(avg_metrics_dict)
        df_avg_metrics.index = ['Epoch {}'.format(i) for i in range(1, len(avg_metrics_dict[self.metrics_str[0]]) + 1)]
        return df_avg_metrics

    def get_models_hist_comparison_df(self, metric='f1'):
        """
            Method that returns a pandas Data Frame where the columns correspond to each one of the trained
            models' history corresponding to the metric specified.

            :param metric       (str)           metric to make the comparison between models.
            :return: hist_df    (Data Frame)    pandas Data Frame with the comparison between models based on the
                                                provided metric.
        """
        if not self.models_history:
            raise ModelError('Cannot obtain comparison Data Frame because no model selection has been performed. '
                             'Run the "model_selection()" method before obtaining the pandas Data Frame.')
        comparison_dict = {}
        for model_name, model_history in self.models_history.items():
            comparison_dict[model_name] = model_history['average_metric_history'][metric]
        comparison_df = pd.DataFrame(data=comparison_dict)
        return comparison_df

    def get_dict_metric_comparison(self):
        """
            Returns a dictionary where the keys correspond to the metric name, and the corresponding value to
            a pandas Data Frame containing the corresponding metrics comparison for all models.

            :return: comparison_dict    (dict)  dictionary with all metric comparisons across all models.
        """
        comparison_dict = {}
        for metric in self.metrics:
            if type(metric) != str:
                metric = metric.name
            comparison_dict[metric] = self.get_models_hist_comparison_df(metric=metric)
            comparison_dict['val_' + metric] = self.get_models_hist_comparison_df(metric='val_' + metric)
        return comparison_dict

    def plot_training_history(self):
        """
            Plots a figure with the training history of the model, throughout all epochs,
            which are averaged over all k folds.
                :return: None
        """

        # Plot model selection
        if self.models_history:
            dv = DataVisualizer(self.get_dict_metric_comparison())
            dv.visualize_multiple_df()
        # Plot single model
        elif self.model_history:
            metrics_df = self.get_average_metrics_hist_df()
            model_hist = {}
            for metric in list(metrics_df.columns):
                df_metric = pd.DataFrame(data={'metric': metrics_df[metric].to_list()})
                model_hist[metric] = df_metric
            dv = DataVisualizer(model_hist)
            dv.visualize_multiple_df()
        else:
            raise ModelError('Cannot visualize a model that has not been trained! '
                             'Run the "model_selection()" or "fit()" methods to train models beforehand.')


class NNClassifier(KerasClassifier):
    """
        The classical multilayer perceptron class is implemented. Children Class from the KerasClassifier.
    """

    def __init__(self,
                 input_shape=None,
                 n_classes=2,
                 n_dense_layers=2,
                 dense_activation='relu',
                 n_dense_units_list=None,
                 dense_initializer='glorot_normal',
                 dense_l1=0,
                 dense_l2=0,
                 metrics=None):

        """
            Constructor of the class. Configures all the neural network parameters to create the model.

                :param input_shape:         (tuple) input tensor shape to be fed to the Neural Network.
                :param n_classes:           (int)   number of classes to be classified.
                                                    Check the documentation of Keras for more details.
                :param n_dense_layers:      (int)   number of dense layers of the network.
                :param dense_activation:    (str)   type of neuron activation to use in the hidden dense layers.
                                                    Check the documentation of Keras for more details.
                :param n_dense_units_list:  (int)   list with length equal to the number of dense layers in the network.
                                                    Each element in the list represents the number of units per layer.
                :param dense_initializer:   (str)   Initialization method to set the dense neuron weights and biases
                                                    before training. Check the documentation of Keras for more details.
                :param dense_l1:            (float) L1 regularization constant for dense layers to avoid over fitting.
                :param dense_l2:            (float) L2 regularization constant for dense layers to avoid over fitting.
                :param metrics:             (list)  list with all Keras metrics. Each element must be a string.
                                                    Valid strings include: 'f1', 'accuracy', 'sens', 'precision'
                :return: None.
        """
        super().__init__(n_classes=n_classes,
                         input_shape=input_shape,
                         dense_activation=dense_activation,
                         n_dense_layers=n_dense_layers,
                         n_dense_units_list=n_dense_units_list,
                         dense_initializer=dense_initializer,
                         dense_l1=dense_l1,
                         dense_l2=dense_l2,
                         metrics=metrics)
        self.type_classifier = 'nn'

    def create_model(self):

        """
            Builds the model with specifications provided in the constructor.
                :return: None
        """

        if self.input_shape is None:
            raise ModelError('Cannot create the model of the Neural Network architecture,'
                             'if no input shape is defined.')

        self.model = models.Sequential()

        # Create input layer
        self.model.add(Dense(units=self.n_dense_units_list[0],
                             input_dim=self.input_shape[1],
                             activation=self.dense_activation,
                             kernel_initializer=self.dense_initializer,
                             kernel_regularizer=self.dense_regularizer))

        # Create hidden layers
        for idx in range(1, self.n_dense_layers):
            self.model.add(Dense(units=self.n_dense_units_list[idx],
                                 activation=self.dense_activation,
                                 kernel_initializer=self.dense_initializer,
                                 kernel_regularizer=self.dense_regularizer))

        if self.n_classes == 2:
            activation = 'sigmoid'
            units = 1
        else:
            activation = 'softmax'
            units = self.n_classes

        self.model.add(Dense(units, activation=activation,
                             kernel_initializer=self.dense_initializer,
                             kernel_regularizer=self.dense_regularizer))


class BuiltCNN(KerasClassifier):
    """
        Children class of the Keras Classifier. This class is used for all CNN Keras Models whose
        architecture topology is already pre-defined and minor changes must be done to train them.
    """

    def __init__(self,
                 input_shape=None,
                 n_classes=2,
                 n_dense_layers=2,
                 dense_activation='relu',
                 n_dense_units_list=None,
                 dense_initializer='glorot_normal',
                 dense_l1=0,
                 dense_l2=0,
                 metrics=None):
        """
            Constructor of the class. Configures all the neural network parameters to create the model.

                :param input_shape:         (tuple) input tensor shape to be fed to the Neural Network.
                :param n_classes:           (int)   number of classes to be classified.
                :param n_dense_layers:      (int)   number of dense layers of the network.
                :param dense_activation:    (str)   type of neuron activation to use in the hidden dense layers.
                :param n_dense_units_list:  (int)   list with length equal to the number of dense layers in the network.
                                                    Each element in the list represents the number of units per layer.
                :param dense_initializer:   (str)   Initialization method to set the dense neuron weights and biases
                                                    before training. Check the documentation of Keras for more details.
                :param dense_l1:            (float) L1 regularization constant for dense layers to avoid over fitting.
                :param dense_l2:            (float) L2 regularization constant for dense layers to avoid over fitting.
                :param metrics:             (list)  list with all Keras metrics. Each element must be a string.
                                                    Valid strings include: 'f1', 'accuracy', 'sens', 'precision'
                :return: None.
        """
        super().__init__(n_classes=n_classes,
                         input_shape=input_shape,
                         dense_activation=dense_activation,
                         n_dense_layers=n_dense_layers,
                         n_dense_units_list=n_dense_units_list,
                         dense_initializer=dense_initializer,
                         dense_l1=dense_l1,
                         dense_l2=dense_l2,
                         metrics=metrics)
        self.type_classifier = 'built_cnn'

    def get_base_model(self):
        """
            Method to be implemented in Children classes.
            It returns the base topology network (i.e. ResNet, InceptionNet, DenseNet, etc.).
                :return: base_model: Keras pre-built CNN.
        """
        return models.Model(input_shape=self.input_shape[1::])

    def create_model(self):
        """
            Builds the model with specifications provided in the constructor.
                :return: None
        """

        if self.input_shape is None:
            raise ModelError('Cannot create the model of Built CNN, if no input shape is defined.')

        # Guarantee that input shape has the three channels added as RGB
        if len(self.input_shape) == 3:
            self.input_shape += (3,)

        if self.n_classes == 2:
            activation = 'sigmoid'
            units = 1
        else:
            activation = 'softmax'
            units = self.n_classes

        # Freeze base model layers
        base_model = self.get_base_model()
        base_model.trainable = False

        # Create complete model
        inputs = Input(shape=self.input_shape[1::])
        x = base_model(inputs, training=False)
        x = Flatten()(x)

        # Create hidden layers
        for idx in range(1, self.n_dense_layers):
            x = Dense(units=self.n_dense_units_list[idx],
                      activation=self.dense_activation,
                      kernel_initializer=self.dense_initializer,
                      kernel_regularizer=self.dense_regularizer)(x)

        # Create output classification layer
        outputs = Dense(units, activation=activation)(x)

        # Assign model attribute
        self.model = Model(inputs, outputs)


class VGG16(BuiltCNN):
    """
        Children class of the BuiltCNN. The topology is the VGG16 CNN.
    """

    def __init__(self,
                 input_shape=None,
                 n_classes=2,
                 n_dense_layers=2,
                 dense_activation='relu',
                 n_dense_units_list=None,
                 dense_initializer='glorot_normal',
                 dense_l1=0,
                 dense_l2=0,
                 metrics=None):
        """
            Constructor of the class. Configures all the neural network parameters to create the model.

                :param input_shape:         (tuple) input tensor shape to be fed to the Neural Network.
                :param n_classes:           (int)   number of classes to be classified.
                                                    Check the documentation of Keras for more details.
                :param n_dense_layers:      (int)   number of dense layers of the network.
                :param dense_activation:    (str)   type of neuron activation to use in the hidden dense layers.
                :param n_dense_units_list:  (int)   list with length equal to the number of dense layers in the network.
                                                    Each element in the list represents the number of units per layer.
                :param dense_initializer:   (str)   Initialization method to set the dense neuron weights and biases
                                                    before training. Check the documentation of Keras for more details.
                :param dense_l1:            (float) L1 regularization constant for dense layers to avoid over fitting.
                :param dense_l2:            (float) L2 regularization constant for dense layers to avoid over fitting.
                :param metrics:             (list)  list with all Keras metrics. Each element must be a string.
                                                    Valid strings include: 'f1', 'accuracy', 'sens', 'precision'
                :return: None.
        """

        super().__init__(n_classes=n_classes,
                         input_shape=input_shape,
                         dense_activation=dense_activation,
                         n_dense_layers=n_dense_layers,
                         n_dense_units_list=n_dense_units_list,
                         dense_initializer=dense_initializer,
                         dense_l1=dense_l1,
                         dense_l2=dense_l2,
                         metrics=metrics)
        self.type_classifier = 'vgg16'

    def get_base_model(self):
        """
            Returns the base topology of the VGG16 CNN.
                :return: base_model: vgg16.VGG16()
        """
        base_model = vgg16.VGG16(include_top=False,
                                 input_shape=self.input_shape[1::],
                                 weights='imagenet')
        return base_model


class ResNet50(BuiltCNN):
    """
        Children class of the BuiltCNN. The topology is the ResNet50 CNN.
    """

    def __init__(self,
                 input_shape=None,
                 n_classes=2,
                 n_dense_layers=2,
                 dense_activation='relu',
                 n_dense_units_list=None,
                 dense_initializer='glorot_normal',
                 dense_l1=0,
                 dense_l2=0,
                 metrics=None):
        """
            Constructor of the class. Configures all the neural network parameters to create the model.

                :param input_shape:         (tuple) input tensor shape to be fed to the Neural Network.
                :param n_classes:           (int)   number of classes to be classified.
                                                    Check the documentation of Keras for more details.
                :param n_dense_layers:      (int)   number of dense layers of the network.
                :param dense_activation:    (str)   type of neuron activation to use in the hidden dense layers.
                :param n_dense_units_list:  (int)   list with length equal to the number of dense layers in the network.
                                                    Each element in the list represents the number of units per layer.
                :param dense_initializer:   (str)   Initialization method to set the dense neuron weights and biases
                                                    before training. Check the documentation of Keras for more details.
                :param dense_l1:            (float) L1 regularization constant for dense layers to avoid over fitting.
                :param dense_l2:            (float) L2 regularization constant for dense layers to avoid over fitting.
                :param metrics:             (list)  list with all Keras metrics. Each element must be a string.
                                                    Valid strings include: 'f1', 'accuracy', 'sens', 'precision'
                :return: None.
        """

        super().__init__(n_classes=n_classes,
                         input_shape=input_shape,
                         dense_activation=dense_activation,
                         n_dense_layers=n_dense_layers,
                         n_dense_units_list=n_dense_units_list,
                         dense_initializer=dense_initializer,
                         dense_l1=dense_l1,
                         dense_l2=dense_l2,
                         metrics=metrics)
        self.type_classifier = 'resnet50'

    def get_base_model(self):
        """
            Returns the base topology of the ResNet50 CNN.
                :return: base_model: resnet50.ResNet50
        """
        base_model = resnet50.ResNet50(include_top=False,
                                       input_shape=self.input_shape[1::],
                                       weights='imagenet')
        return base_model


class InceptionResNetV2(BuiltCNN):
    """
        Children class of the BuiltCNN. The topology is the InceptionResNet CNN.
    """

    def __init__(self,
                 input_shape=None,
                 n_classes=2,
                 n_dense_layers=2,
                 dense_activation='relu',
                 n_dense_units_list=None,
                 dense_initializer='glorot_normal',
                 dense_l1=0,
                 dense_l2=0,
                 metrics=None):
        """
            Constructor of the class. Configures all the neural network parameters to create the model.

                :param input_shape:         (tuple) input tensor shape to be fed to the Neural Network.
                :param n_classes:           (int)   number of classes to be classified.
                :param n_dense_layers:      (int)   number of dense layers of the network.
                :param dense_activation:    (str)   type of neuron activation to use in the hidden dense layers.
                                                    Check the documentation of Keras for more details.
                :param n_dense_units_list:  (int)   list with length equal to the number of dense layers in the network.
                                                    Each element in the list represents the number of units per layer.
                :param dense_initializer:   (str)   Initialization method to set the dense neuron weights and biases
                                                    before training. Check the documentation of Keras for more details.
                :param dense_l1:            (float) L1 regularization constant for dense layers to avoid over fitting.
                :param dense_l2:            (float) L2 regularization constant for dense layers to avoid over fitting.
                :param metrics:             (list)  list with all Keras metrics. Each element must be a string.
                                                    Valid strings include: 'f1', 'accuracy', 'sens', 'precision'
                :return: None.
        """

        super().__init__(n_classes=n_classes,
                         input_shape=input_shape,
                         dense_activation=dense_activation,
                         n_dense_layers=n_dense_layers,
                         n_dense_units_list=n_dense_units_list,
                         dense_initializer=dense_initializer,
                         dense_l1=dense_l1,
                         dense_l2=dense_l2,
                         metrics=metrics)
        self.type_classifier = 'inception_resnet_v2'

    def get_base_model(self):
        """
            Returns the base topology of the InceptionResNetV2 CNN.
                :return: base_model: inception_resnet_v2.InceptionResNetV2
        """
        base_model = inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                           input_shape=self.input_shape[1::],
                                                           weights='imagenet')
        return base_model


class DenseNet(BuiltCNN):
    """
        Children class of the BuiltCNN. The topology is the DenseNet CNN.
    """

    def __init__(self,
                 input_shape=None,
                 n_classes=2,
                 n_dense_layers=2,
                 dense_activation='relu',
                 n_dense_units_list=None,
                 dense_initializer='glorot_normal',
                 dense_l1=0,
                 dense_l2=0,
                 metrics=None):
        """
            Constructor of the class. Configures all the neural network parameters to create the model.

                :param input_shape:         (tuple) input tensor shape to be fed to the Neural Network.
                :param n_classes:           (int)   number of classes to be classified.
                                                    Check the documentation of Keras for more details.
                :param n_dense_layers:      (int)   number of dense layers of the network.
                :param dense_activation:    (str)   type of neuron activation to use in the hidden dense layers.
                :param n_dense_units_list:  (int)   list with length equal to the number of dense layers in the network.
                                                    Each element in the list represents the number of units per layer.
                :param dense_initializer:   (str)   Initialization method to set the dense neuron weights and biases
                                                    before training. Check the documentation of Keras for more details.
                :param dense_l1:            (float) L1 regularization constant for dense layers to avoid over fitting.
                :param dense_l2:            (float) L2 regularization constant for dense layers to avoid over fitting.
                :param metrics:             (list)  list with all Keras metrics. Each element must be a string.
                                                    Valid strings include: 'f1', 'accuracy', 'sens', 'precision'
                :return: None.
        """

        super().__init__(n_classes=n_classes,
                         input_shape=input_shape,
                         dense_activation=dense_activation,
                         n_dense_layers=n_dense_layers,
                         n_dense_units_list=n_dense_units_list,
                         dense_initializer=dense_initializer,
                         dense_l1=dense_l1,
                         dense_l2=dense_l2,
                         metrics=metrics)
        self.type_classifier = 'densenet'

    def get_base_model(self):
        """
            Returns the base topology of the DenseNet121 CNN.
                :return: base_model: densenet.DenseNet121
        """
        base_model = densenet.DenseNet121(include_top=False,
                                          input_shape=self.input_shape[1::],
                                          weights='imagenet')
        return base_model


class InceptionV3(BuiltCNN):
    """
        Children class of the BuiltCNN. The topology is the InceptionV3 CNN.
    """

    def __init__(self,
                 input_shape=None,
                 n_classes=2,
                 n_dense_layers=2,
                 dense_activation='relu',
                 n_dense_units_list=None,
                 dense_initializer='glorot_normal',
                 dense_l1=0,
                 dense_l2=0,
                 metrics=None):
        """
            Constructor of the class. Configures all the neural network parameters to create the model.

                :param input_shape:         (tuple) input tensor shape to be fed to the Neural Network.
                :param n_classes:           (int)   number of classes to be classified.
                :param n_dense_layers:      (int)   number of dense layers of the network.
                :param dense_activation:    (str)   type of neuron activation to use in the hidden dense layers.
                                                    Check the documentation of Keras for more details.
                :param n_dense_units_list:  (int)   list with length equal to the number of dense layers in the network.
                                                    Each element in the list represents the number of units per layer.
                :param dense_initializer:   (str)   Initialization method to set the dense neuron weights and biases
                                                    before training. Check the documentation of Keras for more details.
                :param dense_l1:            (float) L1 regularization constant for dense layers to avoid over fitting.
                :param dense_l2:            (float) L2 regularization constant for dense layers to avoid over fitting.
                :param metrics:             (list)  list with all Keras metrics. Each element must be a string.
                                                    Valid strings include: 'f1', 'accuracy', 'sens', 'precision'
                :return: None.
        """

        super().__init__(n_classes=n_classes,
                         input_shape=input_shape,
                         dense_activation=dense_activation,
                         n_dense_layers=n_dense_layers,
                         n_dense_units_list=n_dense_units_list,
                         dense_initializer=dense_initializer,
                         dense_l1=dense_l1,
                         dense_l2=dense_l2,
                         metrics=metrics)
        self.type_classifier = 'inceptionv3'

    def get_base_model(self):
        """
            Returns the base topology of the InceptionV3 CNN.
                :return: base_model: inception_v3.InceptionV3
        """
        base_model = inception_v3.InceptionV3(include_top=False,
                                              input_shape=self.input_shape[1::],
                                              weights='imagenet')
        return base_model


class AlexNet(BuiltCNN):
    """
        Children class of the BuiltCNN. The topology is the AlexNet CNN.
    """

    def __init__(self,
                 input_shape=None,
                 n_classes=2,
                 n_dense_layers=2,
                 dense_activation='relu',
                 n_dense_units_list=None,
                 dense_initializer='glorot_normal',
                 dense_l1=0,
                 dense_l2=0,
                 metrics=None):
        """
            Constructor of the class. Configures all the neural network parameters to create the model.

                :param input_shape:         (tuple) input tensor shape to be fed to the Neural Network.
                :param n_classes:           (int)   number of classes to be classified.
                :param n_dense_layers:      (int)   number of dense layers of the network.
                :param dense_activation:    (str)   type of neuron activation to use in the hidden dense layers.
                                                    Check the documentation of Keras for more details.
                :param n_dense_units_list:  (int)   list with length equal to the number of dense layers in the network.
                                                    Each element in the list represents the number of units per layer.
                :param dense_initializer:   (str)   Initialization method to set the dense neuron weights and biases
                                                    before training. Check the documentation of Keras for more details.
                :param dense_l1:            (float) L1 regularization constant for dense layers to avoid over fitting.
                :param dense_l2:            (float) L2 regularization constant for dense layers to avoid over fitting.
                :param metrics:             (list)  list with all Keras metrics. Each element must be a string.
                                                    Valid strings include: 'f1', 'accuracy', 'sens', 'precision'
                :return: None.
        """

        super().__init__(n_classes=n_classes,
                         input_shape=input_shape,
                         dense_activation=dense_activation,
                         n_dense_layers=n_dense_layers,
                         n_dense_units_list=n_dense_units_list,
                         dense_initializer=dense_initializer,
                         dense_l1=dense_l1,
                         dense_l2=dense_l2,
                         metrics=metrics)
        self.type_classifier = 'alexnet'

        # TODO: implement AlexNet Keras Model
