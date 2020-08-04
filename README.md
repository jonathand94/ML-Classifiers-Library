# ML-Classifiers-Library
The main objective of this library is to standardize the implementation of Machine Learning algorithms for classification (binary or multi-class). Thus, efforts focus on integrating different libraries into a single one to enhance the Data Scientist workflow for any classification task. 

Up until now, only the Keras module has been included to use Deep Learning. Even though, many other modules, such as the Sci-Kit Learn library, will be incorporated to include classic machine learning algorithms such as SVM, Random Forests, Naive Bayes Classifier, etc. 

Right now, other tools have been included, such as visualization tools using Plotly and model comparison tables using Pandas. 

## Basic Usage

For example, to create a Neural Network classifier implement the following line of code:

```
classifier = NNClassifier(**classifier_params)
```

The classifier parameters include:

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
                                                    
You can decide which training parameters to use during training:

                :param x:                   (np_array)      input data with shape [n_samples, **dims]
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

For example, we will train for the MNIST data set

```
import tensorflow as tf
from classifiers import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

training_params = {'x':x,
                   'y':y,
                   'batch_size': 1,
                   'epochs': 30,
                   'k_folds': 10,
                   'lr': 0.001,
                   'optimizer_name': 'adam',
                   'validation_split': 0.7,
                   'shuffle': True,
                   'save_path': save_path,
                   'verbose': 2}
                   
classifier.fit(**training_params)
```

The folder you select for saving the results will include many CSV files with the training and validation process. 

Other features that include this library are: pre-processors, transformers (augmenting data) that you can use seamlessly to integrate within the training. Furthermore, Keras generators were included for training with large data sets. 

## Pre-processors

As mentioned above, one feature consists on creating pre-processor objects to add to the training process (Interestingly, Chest X-Ray pre-processors were also added for segmenting lungs). Here is an example where principal components and Hog Features are extracted before training:

```
from pre_processors import ImagePreProcessor
from classifiers import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

pre_processor_params = {'flatten_flag': True,
                        'extract_hog_flag': True,
                        'n_pca_components': 10,
                        'verbose': 0}
pre_processor = ImagePreProcessor(**pre_processor_params)

training_params = {'x':x,
                   'y':y,
                   'batch_size': 1,
                   'epochs': 30,
                   'k_folds': 10,
                   'lr': 0.001,
                   'optimizer_name': 'adam',
                   'validation_split': 0.7,
                   'shuffle': True,
                   'pre_processor': pre_processor,
                   'save_path': save_path,
                   'verbose': 2}
             
classifier.fit(**training_params)
```

## Transformers

You can add transformers to augment the data during training.

```
from transformers import ImageTransformerClassifier
from classifiers import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

transformer_params = {'degree_range': 10,
                      'n_rotations': 1,
                      'horizontal_shift': 0.05,
                      'vertical_shift': 0.05,
                      'n_translations': 0,
                      'zoom_range': (0.85, 0.85),
                      'n_zooms': 0,
                      'brightness_range': (0, 0.2),
                      'n_brightness_shifts': 0,
                      'contrast_range': (0.8, 1),
                      'n_contrast_shifts': 0,
                      'flip_left_right': False,
                      'flip_up_down': False,
                      'shuffle': True}
transformer = ImageTransformerClassifier(**transformer_params)

training_params = {'x':x,
                   'y':y,
                   'batch_size': 1,
                   'epochs': 30,
                   'k_folds': 10,
                   'lr': 0.001,
                   'optimizer_name': 'adam',
                   'validation_split': 0.7,
                   'shuffle': True,
                   'transformer': transformer,
                   'save_path': save_path,
                   'verbose': 2}
             
classifier.fit(**training_params)
```

## Model Selection

Model Selection is an important part when training classifiers and selecting the best one. Thus, this feature was included as a simple method for the Classifier object. Random selection was implemented, but Bayesian search using the TPE algorithm will be included in the future.

First you need to define your search spaces: 

```
lr_space = -5
batch_size_space = [16, 32]
n_layers_space = list(range(3, 5, 1))
n_units_space = list(range(20, 200, 20))
regularizer_space = -5
hidden_activation_space = ['relu', 'elu', 'tanh']
initializer_space = ['glorot_normal', 'glorot_uniform']
optimizer_space = ['Adam', 'SGD', 'RMSprop']

hyper_parameter_space = {'lr': lr_space,
                         'batch_size': batch_size_space,
                         'optimizer': optimizer_space,
                         'n_dense_layers': n_layers_space,
                         'dense_units': n_units_space,
                         'dense_regularizer': regularizer_space,
                         'dense_activation': hidden_activation_space,
                         'dense_initializer': initializer_space}
```

Then, you need to specify the parameters of the model selection process:

```
model_selection_params = {'x': x,
                          'y': y, 
                          'target_metric': 'f1',
                          'search_mode': 'random',
                          'n_models': 5,
                          'hyper_parameter_space': hyper_parameter_space,
                          'epochs': 15,
                          'k_folds': 10,
                          'validation_split': 0.7,
                          'shuffle': True,
                          'pre_processor': pre_processor,
                          'transformer': transformer,
                          'save_path': save_path,
                          'verbose': 2}
```

Finally, you run the method of model selection:

```
classifier.model_selection(**model_selection_params)
```

## Loading and Saving the Classifier

If you want to save your classifier and keep training it. You can do as follows:

```
from utils import FileManager

classifier.reset_keras_models()
file_manager = FileManager(file=classifier)
file_manager.save_file(save_path=save_path + '/{}.pickle'.format('neural_network'))

file_manager = FileManager()
classifier = file_manager.load_file(file_path=save_path + '/{}.pickle'.format(type_classifier))
classifier.load_model(model_path=save_path + '/best_model')
classifier.fit(x=x_test, y=y:test, epochs=100)
```


                                                 
