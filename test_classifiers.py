from classifiers import *
from transformers import ImageTransformerClassifier, TransformerClassifier
from pre_processors import ChestXRayPreProcessor
import os
import plotly

# -------------------------------------------------- INPUTS -----------------------------------------------------

# -------------------------- General --------------------------

test_fit = False
test_fit_generator = False

test_model_selection = True
use_generator_model_selection = False

test_classifier_performance = True

# 'nn', 'inceptionv3', 'densenet', 'inception_resnet_v2', 'resnet50', 'vgg16'
type_classifier = 'inception_resnet_v2'

# The input shape must match the input received by the classifier
# Thus, you need to take into account how the shape of the data will look like
# after pre_processing it
input_shape = (256, 256, 3)

# Shape to resize all image data
image_shape = (256, 256)

training_path = 'covid19_repo/data/x_rays/covid'
save_path = 'Models/{}/No UNet/Model Selection (256, 256) CPU'.format(type_classifier)

# ---------------------- Pre-Processing -----------------------

u_net_model_path = 'covid19_repo/u_net_model/Unet_model.json'
u_net_weights_path = 'covid19_repo/u_net_model/Unet_model.h5'

if type_classifier == 'nn':
    pre_processor_params = {'flatten_flag': True,
                            'extract_hog_flag': False,
                            'segment_lungs_flag': False,
                            'segment_lungs_method': 'u_net',
                            'u_net_model_path': u_net_model_path,
                            'u_net_weights_path': u_net_weights_path,
                            'u_net_batch_size': 8,
                            'u_net_threshold': 0.1,
                            'n_pca_components': 10,
                            'verbose': 0}
    pre_processor = ChestXRayPreProcessor(**pre_processor_params)
else:
    pre_processor_params = {'type_cnn': type_classifier,
                            'extract_hog_flag': False,
                            'segment_lungs_flag': False,
                            'segment_lungs_method': 'u_net',
                            'u_net_model_path': u_net_model_path,
                            'u_net_weights_path': u_net_weights_path,
                            'u_net_batch_size': 4,
                            'u_net_threshold': 0.1,
                            'verbose': 0}
    pre_processor = ChestXRayPreProcessor(**pre_processor_params)

# --------------------- Data Augmentation ----------------------

if type_classifier == 'nn':
    transformer_params = {'smote_params': {'random_state': None, 'k_neighbors': 5},
                          'shuffle': True}
    transformer = TransformerClassifier(**transformer_params)
else:
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

# ------------------------- Classifier --------------------------

metrics = ['accuracy', 'f1', 'sensitivity', 'auc', 'specificity', 'precision']
classifier_params = {'n_dense_layers': 2,
                     'n_dense_units_list': [100, 50],
                     'dense_activation': 'relu',
                     'dense_initializer': 'glorot_normal',
                     'dense_l1': 0.001,
                     'dense_l2': 0.001,
                     'metrics': metrics}

# ------------------------- Classic Fit -------------------------
training_params = {}
if test_fit or test_fit_generator:
    training_params = {'batch_size': 1,
                       'epochs': 30,
                       'k_folds': 10,
                       'lr': 0.001,
                       'optimizer_name': 'adam',
                       'validation_split': 0.7,
                       'shuffle': True,
                       'pre_processor': pre_processor,
                       'transformer': transformer,
                       'save_path': save_path,
                       'verbose': 2}

# ------------------------ Model Selection ------------------------

if test_model_selection:
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

    model_selection_params = {'target_metric': 'f1',
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

# -------------------------- Generator --------------------------

generator_params = {'batch_size': 4,
                    'transformer': transformer,
                    'pre_processor': pre_processor,
                    'shuffle': True,
                    'verbose': 2}


# ----------------------------------------------------- DATA -----------------------------------------------------

if test_fit_generator or (use_generator_model_selection and test_model_selection):
    pos_training_data_path = training_path + '/positive_covid/'
    neg_training_data_path = training_path + '/negative_covid/'

    # Get valid image extensions
    valid_img_extensions = ['jpg', 'jpeg', 'png', 'bmp']

    #  Getting file names from folders
    pos_x_files = os.listdir(pos_training_data_path)
    pos_x_files = [img for img in pos_x_files if img.split('.')[-1] in valid_img_extensions]
    neg_x_files = os.listdir(neg_training_data_path)
    neg_x_files = [img for img in neg_x_files if img.split('.')[-1] in valid_img_extensions]

    # Crate dictionary with labels.
    #       Keys: class type (i.e. '0', '1', '2', etc.)
    #       Values: file names (i.e. 'filename1', 'filename2', etc)
    pos_x_file_paths = [pos_training_data_path + file for file in pos_x_files]
    y = {file: 1 for file in pos_x_file_paths}
    neg_x_file_paths = [neg_training_data_path + file for file in neg_x_files]
    y.update({file: 0 for file in neg_x_file_paths})

    # List with file names of each training sample
    #       (i.e. ['image1.jpg', 'image2.bmp', 'image3.png'])
    x = pos_x_file_paths
    x.extend(neg_x_file_paths)

    # Update classifier parameters
    classifier_params['n_classes'] = len(set(y.values()))

    # Update generator parameters
    generator_params['input_shape'] = (len(x),) + input_shape
    generator_params['image_shape'] = image_shape
    generator_params['x_file_paths'] = x
    generator_params['labels_dict'] = y
else:
    fm = FileManager()
    x, y = fm.get_classifier_image_training_data(training_data_path=training_path, shape=image_shape)
    classifier_params['n_classes'] = len(np.unique(y))

# ------------------------------------------------- CLASSIFIER --------------------------------------------------

classifier = None

# Updating classifier parameters
if type(input_shape) == int:
    classifier_params['input_shape'] = (len(x), input_shape)
else:
    classifier_params['input_shape'] = (len(x),) + input_shape

if type_classifier == 'vgg16':
    classifier = VGG16(**classifier_params)
elif type_classifier == 'resnet50':
    classifier = ResNet50(**classifier_params)
elif type_classifier == 'inception_resnet_v2':
    classifier = InceptionResNetV2(**classifier_params)
elif type_classifier == 'densenet':
    classifier = DenseNet(**classifier_params)
elif type_classifier == 'inceptionv3':
    classifier = InceptionV3(**classifier_params)
elif type_classifier == 'nn':
    classifier = NNClassifier(**classifier_params)


# ----------------------------------------------------- TRAINING -----------------------------------------------------

if test_fit_generator:
    generator = generators.ClassifierImageGenerator(**generator_params)
    training_params['generator'] = generator
    classifier.fit_generator(**training_params)


if test_fit:
    training_params['x'] = x
    training_params['y'] = y
    classifier.fit(**training_params)


if test_model_selection:
    model_selection_params['x'] = x
    model_selection_params['y'] = y

    if use_generator_model_selection:
        generator = generators.ClassifierImageGenerator(**generator_params)
        model_selection_params['x'] = generator
    classifier.model_selection(**model_selection_params)


if save_path:
    # Save
    classifier.reset_keras_models()
    file_manager = FileManager(file=classifier)
    file_manager.save_file(save_path=save_path + '/{}.pickle'.format(type_classifier))

    # Load
    file_manager = FileManager()
    classifier = file_manager.load_file(file_path=save_path + '/{}.pickle'.format(type_classifier))
    classifier.load_model(model_path=save_path + '/best_model')
    classifier.fit_generator(x=generator, y=None, epochs=100)



# ----------------------------------------------------- TESTING -----------------------------------------------------

if test_classifier_performance:
    # Load test data
    fm = FileManager()
    x, y = fm.get_classifier_image_training_data(training_data_path=training_path, shape=image_shape)
    x_test = x[0:500]
    y_test = y[0:500]

    # Visualize test performance
    print('Binary metrics: {}'.format(classifier.get_binary_metrics(x_test=x_test, y_test=y_test)))
    confusion_matrix_fig = classifier.get_confusion__matrix_fig(x_test=x_test, y_test=y_test)
    plotly.offline.plot(confusion_matrix_fig)
    roc_curve_fig = classifier.get_roc_curve_fig(x_test=x_test, y_test=y_test)
    roc_curve_fig.show()


