import os
import numpy as np
from utils import DicomFileManager, FileManager, BinaryClassifierComparator
from visualizers import DataVisualizer
import plotly


if __name__ == "__main__":

    test_data_visualizer = False
    test_pre_processor = False
    test_dicom_file_manager = False
    test_file_manager = False
    test_classifier_comparator = True

    # ----------------------------------- FileManager -----------------------------------

    if test_file_manager:
        fm = FileManager()
        training_path = 'covid19_repo/data/x_rays/covid'
        shape = (64, 64)
        shuffle = True
        x, y = fm.get_classifier_image_training_data(training_data_path=training_path,
                                                     shape=shape,
                                                     shuffle=shuffle)
        print(x.shape, y.shape)

    # ----------------------------------- DataVisualizer -----------------------------------

    if test_data_visualizer:

        visualize_pca = False
        visualize_cnn_activations = True

        # ---------- Visualizing PCA ---------
        if visualize_pca:
            n_classes = 3
            x = np.random.random((100, 32, 32))
            y = np.random.randint(0, n_classes, size=100)
            dv = DataVisualizer(x=x, y=y)
            n_pca_components = 3
            dv.visualize_pca(n_pca_components=n_pca_components)

        # ---- Visualizing CNN Activations ----

        if visualize_cnn_activations:
            # Load CNN to visualize layer activations
            cnn_path = 'Models/CNN/VGG16/Classic Training/Model 1'
            file_manager = FileManager()
            cnn = file_manager.load_file(file_path=cnn_path + '/nn.pickle')
            cnn.load_model(model_path=cnn_path + '/model')

            # Get just few samples data from which the CNN trained
            x = cnn.data[0:2]  # We are visualizing two training samples
            dv = DataVisualizer(x=x)

            # Visualize layers, usually we want the last one
            layers = [-1]
            dv.visualize_activations(model=cnn.model, layers=layers)

    # ----------------------------------- DicomFileManager -----------------------------------

    if test_dicom_file_manager:

        # ---------------- Visualize Dicom Files in a folder ----------------

        dfm = DicomFileManager()
        folder_path = 'covid19_repo/data/x_rays/covid_data/'

        fm = DicomFileManager()
        for file in os.listdir(folder_path):
            if file.endswith('.dcm'):
                fm.load_file(file_path=folder_path + file)
                im = fm.get_image_data()
                im.show(title=file)

        # ---------------- Save Dicom Files into a folder as images ----------------

        dfm = DicomFileManager()
        folder_path = 'covid19_repo/data/x_rays/covid_data/'
        save_path = 'covid19_repo/'
        img_extension = '.jpg'

        fm = DicomFileManager()
        for file in os.listdir(folder_path):
            if file.endswith('.dcm'):
                fm.load_file(file_path=folder_path + file)
                fm.save_pixels_as_image(save_path=save_path + file.replace('.dcm', img_extension))

    # ----------------------------------- BinaryClassifierComparator -----------------------------------

    if test_classifier_comparator:

        pos_data_path = 'test data set/test_positive_covid'
        neg_data_path = 'test data set/test_negative_covid'
        models_folder = 'COVID19 Best Detection Models/'

        fm = FileManager()

        print('Getting all training data...')
        data_path = 'test data set'
        shape = (256, 256)
        shuffle = False
        x, y = fm.get_classifier_image_training_data(training_data_path=data_path,
                                                     shape=shape,
                                                     shuffle=shuffle)
        data_dict = {'image_{}'.format(i): [np.expand_dims(x[i], axis=0), y[i]] for i in range(len(x))}

        classifier_dict = {}
        model_names = [folder for folder in os.listdir(models_folder) if '.' not in folder]
        for model_name in model_names:
            print('\rLoading classifier: {}'.format(model_name, end=''))
            model_path = models_folder + model_name
            classifier_dict[model_name] = fm.load_file(model_path + '/{}.pickle'.format(model_name))
            classifier_dict[model_name].load_model(model_path=model_path + '/model')

            print('Plotting confusion matrix')
            confusion_matrix_fig = classifier_dict[model_name].get_confusion__matrix_fig(x_test=x, y_test=y)
            plotly.offline.plot(confusion_matrix_fig)

        # ---------------- Get comparison DF ----------------

        comparator = BinaryClassifierComparator(classifiers_dict=classifier_dict,
                                                test_data_dict=data_dict)
        predictions = comparator.predict_samples(threshold=0.5)

        # TODO: check resource exhaustion
        # Get results
        print('Computing comparison DF...')
        comparison_df = comparator.get_comparison_df()
        comparison_df.to_csv('COVID19 Best Detection Models/classifier_comparison_excluded.csv', index=False)

        # ------------- Get Binary Metrics DF -------------
        print('Getting binary metrics.')
        binary_metrics_df = comparator.get_binary_metrics_df(x_test=x, y_test=y)
        binary_metrics_df.to_csv('COVID19 Best Detection Models/binary_metrics_excluded.csv', index=False)




