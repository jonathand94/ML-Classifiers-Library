from generators import ClassifierImageGenerator
from transformers import ImageTransformerClassifier
from pre_processors import ChestXRayPreProcessor
from visualizers import DataVisualizer
import os

# ------------------------------ INPUTS ------------------------------

pos_training_data_path = 'covid19_repo/data/x_rays/covid/positive_covid/'
neg_training_data_path = 'covid19_repo/data/x_rays/covid/negative_covid/'
data_shape = (512, 512, 3)

u_net_model_path = 'covid19_repo/u_net_model/Unet_model.json'
u_net_weights_path = 'covid19_repo/u_net_model/Unet_model.h5'
pre_processor_params = {'type_cnn': '',
                        'n_pca_components': 0,
                        'extract_hog_flag': False,
                        'segment_lungs_flag': True,
                        'segment_lungs_method': 'u_net',
                        'u_net_model_path': u_net_model_path,
                        'u_net_weights_path': u_net_weights_path,
                        'u_net_batch_size': 8,
                        'u_net_threshold': 0.1,
                        'verbose': 0}
pre_processor = ChestXRayPreProcessor(**pre_processor_params)

transformer_params = {'degree_range': 10,
                      'n_rotations': 1,
                      'horizontal_shift': 0.05,
                      'vertical_shift': 0.05,
                      'n_translations': 1,
                      'zoom_range': (0.85, 0.85),
                      'n_zooms': 1,
                      'brightness_range': (0, 0.2),
                      'n_brightness_shifts': 1,
                      'contrast_range': (0.8, 1),
                      'n_contrast_shifts': 1,
                      'flip_left_right': True,
                      'flip_up_down': False,
                      'shuffle': True}
transformer = ImageTransformerClassifier(**transformer_params)

generator_params = {'batch_size': 80,
                    'shuffle': True,
                    'verbose': 2,
                    'pre_processor': pre_processor,
                    'transformer': transformer}


# ------------------------------ DATA ------------------------------

#  Getting file names from folders
pos_x_files = os.listdir(pos_training_data_path)
neg_x_files = os.listdir(neg_training_data_path)

pos_x_file_paths = [pos_training_data_path + file for file in pos_x_files]
y = {file: 1 for file in pos_x_file_paths}

neg_x_file_paths = [neg_training_data_path + file for file in neg_x_files]
y.update({file: 0 for file in neg_x_file_paths})

x = pos_x_file_paths
x.extend(neg_x_file_paths)

# Updating generator parameters
generator_params['input_shape'] = (len(x),) + data_shape
generator_params['x_file_paths'] = x
generator_params['labels_dict'] = y


# -------------------------- GENERATOR --------------------------

# Generator function
def gen():
    for item in ClassifierImageGenerator(**generator_params):
        yield item


# Generate data
sum_dt = 0
n_iter = 1
for i,  (x, y) in enumerate(gen()):
    print('Generated data of shape: {}'.format(x.shape))
    print('Generated labels of shape: {}'.format(y.shape))
    print('-------------------------------------------------------')
    if i == n_iter-1:
        break


# ------------------------ VISUALIZATION ------------------------

# Visualizing generated data
dv = DataVisualizer()
dv.plot_images(images=x[40:80], image_names=y[40:80], title='Generated')
