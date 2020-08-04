import tensorflow as tf
from tensorflow.keras import models
from skimage.transform import resize
from matplotlib import pyplot as plt
import math
from sympy import factorint
import matplotlib
from pre_processors import PreProcessor
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import warnings
import dash
import plotly.graph_objs as go
import plotly
import numpy as np
from errors import DimensionalityError
import pandas as pd


class DataVisualizer:
    """
        Class that allows visualizing multiple things of the data.
    """

    def __init__(self, x=None, y=np.array([])):

        """
            Construct the object with the data "x" to be visualized, while a vector of labels "y" can be defined
            for visualizing principal components.

                :param x:   [data]      data for visualizing (can be a numpy array, a pandas Data Frame, etc.)
                :param y:   [np_array]  an optional vector of labels with shape [n_samples, 1]
                                        (applicable for visualizing PCA)
        """
        self.x = x
        self.y = y

    @staticmethod
    def plot_images(images, title='Images', figsize=8, dpi=100, image_names=None):
        """
            Plots a list of images in a subplot that adapts its shape dynamically based
            on the number of images to be plotted.

                :param images:      (np_array)      array with all images to be plotted.
                                                    Its shape could be: (n_images, img_width, img_height, n_channels),
                                                    or (n_images, img_width, img_height)
                :param title:       (str)           Title of the figure.
                :param figsize      (float, float)  width, height in inches.
                :param dpi:         (int)           resolution of the figure.
                :param image_names  (list of str)   optional list if you want to assign different titles to each image

            :return: None
        """

        if image_names is None:
            image_names = ['Image {}'.format(i) for i in range(1, len(images)+1)]

        if len(image_names) != len(images):
            raise ValueError('The number image names {} does not match '
                             'the number of images {}.'.format(len(images),
                                                               len(image_names)))

        matplotlib.rc('font', size=8, weight='bold')

        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        title_obj = plt.title(title, x=0.5, y=1)
        plt.setp(title_obj, color='r')

        factors_dictionary = factorint(len(images), limit=100)

        prime_factors = []
        for factor in list(factors_dictionary.keys()):
            prime = int(factor)
            repetitions = factors_dictionary[factor]
            for i in range(repetitions):
                prime_factors.append(prime)

        if len(prime_factors) == 1:  # It is a prime
            columns = np.ceil(np.sqrt(len(images)))
            rows = np.ceil(len(images) / columns)
        else:  # Multiple primes (big number)
            columns = np.product(np.array(prime_factors[0:int(len(prime_factors)/2)+1]))
            rows = np.product(prime_factors[int(len(prime_factors)/2)+1::])

        for i in range(0, len(images)):

            # Setting figure attributes
            axs = fig.add_subplot(rows, columns, i+1)
            title_obj = axs.set_title(image_names[i], x=0.5, y=0.5)
            plt.setp(title_obj, color='r')
            plt.imshow(images[i])
            plt.axis('off')
        plt.show()

    def visualize_pca(self, n_pca_components=2):

        """
            Method that allows the visualization of the principal components of the data.

                :param n_pca_components: [int]  number of principal components to graph from the data.
                                                Must be an integer between 2 and 3.
                :return: None.
        """

        if type(self.x) != np.ndarray:
            raise TypeError('TypeError: only numpy arrays are admitted for the matrix "x"!')
        if type(self.y) != np.ndarray:
            raise TypeError('TypeError: only numpy arrays are admitted for the vector "y"! ')
        if len(self.y.shape) > 2 and self.y is not None:
            raise DimensionalityError(
                'DimensionalityError: Invalid shape for the "x" matrix! A shape {0} was provided, '
                'but admissible shapes are: (n_samples,) and (n_samples, 1)'.format(self.x.shape))
        if len(self.y.shape) == 2 and 1 not in self.y.shape and self.y is not None:
            raise DimensionalityError(
                'DimensionalityError: Invalid shape for the "x" matrix! A shape {0} was provided, '
                'but admissible shapes are: (n_samples,) and (n_samples, 1)'.format(self.x.shape))
        if self.y.shape[0] != self.x.shape[0] and self.y is not None:
            raise DimensionalityError(
                'DimensionalityError: the number of samples "x" in the matrix  differ from the number '
                'of samples in the vector of labels "y". They must be equal!'.format(self.x.shape))

        # Reshape if needed
        if len(self.x.shape) != 2:
            warnings.warn('Data was reshaped to {} to visualize PCA'.format(self.x.shape))
            self.x = np.reshape(self.x, (self.x.shape[0], np.product(self.x.shape[1::])))

        if n_pca_components != 2 and n_pca_components != 3:
            raise ValueError('Specify a valid number of principal components to visualize! '
                             'The value must be an integer: either 2 or 3.')

        # Normalize and compute PCA
        pr = PreProcessor()
        x = pr.pca(x=self.x, n_pca_components=n_pca_components)

        # Initialize figure
        pca_fig = go.Figure()

        # Create colors if labels were specified
        if self.y is not None:
            labels = list(set(list(self.y)))

            # Separate all classes and create Figure
            for label in labels:

                # Assign random color per class
                r, g, b = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
                color = f'rgb({r}, {g}, {b})'

                # Get indices where the class is located
                idx = self.y == label

                # Add scatter to figure
                if n_pca_components == 2:
                    pca_fig.add_trace(go.Scatter(x=x[idx, 0],
                                                 y=x[idx, 1],
                                                 mode='markers',
                                                 name='Class {}'.format(label),
                                                 marker=dict(color=color, size=10)))

                elif n_pca_components == 3:
                    pca_fig.add_trace(go.Scatter3d(x=x[idx, 0],
                                                   y=x[idx, 1],
                                                   z=x[idx, 2],
                                                   mode='markers',
                                                   name='Class {}'.format(label),
                                                   marker=dict(color=color, size=10)))

        # No labels were specified
        else:
            if n_pca_components == 2:
                pca_fig.add_trace(go.Scatter(x=x[:, 0],
                                             y=x[:, 1],
                                             mode='markers'))
            elif n_pca_components == 3:
                pca_fig.add_trace(go.Scatter3d(x=x[:, 0],
                                               y=x[:, 1],
                                               z=x[:, 2],
                                               mode='markers'))

        # Create layout
        if n_pca_components == 2:
            pca_fig.update_layout(title='<b>Principal Component Analysis</b>',
                                  title_x=0.5,
                                  xaxis_title='<b>Principal Component 1</b>',
                                  yaxis_title='<b>Principal Component 2</b>')
        elif n_pca_components == 3:
            pca_fig.update_layout(title='<b>Principal Component Analysis</b>',
                                  title_x=0.5,
                                  scene=dict(xaxis_title='<b>Principal Component 1</b>',
                                             yaxis_title='<b>Principal Component 2</b>',
                                             zaxis_title='<b>Principal Component 3</b>')
                                  )
        plotly.offline.plot(pca_fig)

    @staticmethod
    def get_scatter_df_fig(df,
                           fig_titile='Scatter Plot',
                           xaxis_title='x',
                           yaxis_title='y'):

        """
            Plots different scatter plots, each one based on the data contained at each column
            of the pandas Data Frame.
                :param df:              (Data Frame)        pandas Data Frame.
                :param fig_titile:      (str)               name of the scatter plot figure.
                :param xaxis_title:     (str)               title for the x axis.
                :param yaxis_title:     (str)               title for the y axis.
                :return:                (plotly graph obj)  scatter plot with all pandas data frame information.
        """

        if type(df) is not pd.core.frame.DataFrame:
            raise TypeError('Only pandas Data Frames are accepted as arguments!')

        if not list(df.columns):
            raise ValueError('Cannot create scatter plot with an empty pandas Data Frame!')

        # Initialize figure
        scatter_fig = go.FigureWidget()

        # Create scatter for each column
        for column in df.columns:
            y_data = df[column].to_list()
            x_data = list(range(1, len(y_data)))
            scatter_fig.add_trace(go.Scatter(x=x_data,
                                             y=y_data,
                                             mode='lines',
                                             name=column))

        scatter_fig.update_layout(title='<b>' + fig_titile + '</b>',
                                  title_x=0.5,
                                  xaxis_title='<b>' + xaxis_title + '</b>',
                                  yaxis_title='<b>' + yaxis_title + '</b>'
                                  )
        return scatter_fig

    @staticmethod
    def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
        # Print New Line on Complete
        if iteration == total:
            print()

    def visualize_multiple_df(self):
        """
            Method that creates a visualization app to visualize multiple Data Frames contained in a dictionary.
            Each key will be a different plot, while each value corresponds to the Data Frame to be plot.
            In each plot, different scatters are added corresponding to all columns included in the corresponding
            Data Frame.
            :return: None
        """
        if type(self.x) is not dict:
            raise TypeError('Only dictionaries are accepted as arguments!')

        # Creating visualization app
        external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
        external_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash('model_visualization',
                        external_scripts=external_js,
                        external_stylesheets=external_css)

        # Get options of drop down
        plot_ids = list(self.x.keys())
        options_drop_down = [({'label': plot_id, 'value': plot_id}) for plot_id in plot_ids]

        app.layout = html.Div(children=[
            html.Div(children='''
                Enter value to be visualize the plot:
            '''),
            dcc.Dropdown(
                id='input_drop_down',
                options=options_drop_down,
                value=plot_ids[0]
            ),
            dcc.Graph(id='df_plot'),
        ])

        @app.callback(
            Output(component_id='df_plot', component_property='figure'),
            [Input(component_id='input_drop_down', component_property='value')]
        )
        def update_value(plot_id):
            df_fig = self.get_scatter_df_fig(df=self.x[plot_id],
                                             fig_titile=plot_id.upper(),
                                             xaxis_title='Epochs',
                                             yaxis_title=plot_id.upper())
            return df_fig

        app.run_server(debug=False)

    # TODO: implement HOG features visualizer

    def visualize_activations(self, model, layers=None):

        """
            Method that allows the visualization of the activations of a keras model layers.
            It takes the "x" data from the constructor and requires a model to test
            By default, it will show a figure for each data sample with the activations of the last layer
            If a list of layers index is provided, it will show a figure for each layer requested with all data sample

                :param model:   [keras_model]   keras model whose activations will be shown
                :param layers:  [list of int]   indicate the index of the layers whose activations will be shown
                                                if empty, it only shows the last layer
                :return:        [np.ndarray or list]
        """

        if type(self.x) != np.ndarray:
            raise TypeError('TypeError: only numpy arrays are admitted for the matrix "x"!')

        if type(model) is not tf.keras.Model:
            raise TypeError('The model is not a valid tf.keras.Model!')

        # Choose last layer
        if not layers:
            layers = [-1]

        x = self.x
        #   Select the layers whose activations will be obtained
        layers_output = [model.layers[layer_indx].output for layer_indx in layers]
        #   Create a new model to get the activations from the layers of interest
        activation_model = models.Model(inputs=model.input, outputs=layers_output)
        #   Generate the activations for each sample in the input data
        activations = activation_model.predict(x)

        if type(activations) is np.ndarray:
            # It's a numpy array
            # It's a single layer output

            if len(activations.shape) < 3:
                # It's a dense layer output process as such

                l, b = activations.shape

                n_rows = round(np.sqrt(l))
                n_cols = math.ceil(np.sqrt(l))

                plt.figure(num=0, figsize=(15 * 2, 8))

                for imn in range(l):
                    actvtns = np.reshape(activations[imn], (1, b))

                    plt.subplot(n_rows, n_cols, imn + 1)
                    plt.title('Image ' + str(imn))
                    plt.imshow(actvtns)
                    plt.colorbar()
                    plt.clim(0, 1)

                plt.suptitle('This are the activations for layer ' + str(model.layers[layers[0]].name), fontsize=14)
                # plt.show()

            else:
                # It's a convolutional layer output, process as such
                l, w, h, b = activations.shape

                n_rows = round(np.sqrt(l))
                n_cols = math.ceil(np.sqrt(l))

                plt.figure(num=0, figsize=(15 * 2, 8))

                for imn in range(l):
                    acts_sum = np.zeros((w, h))
                    for n in range(b):
                        acts_sum = acts_sum + activations[imn, :, :, n]

                    acts_sum *= 1.0 / acts_sum.max()

                    resized = resize(acts_sum, (x[imn].shape[0], x[imn].shape[1]))

                    plt.subplot(n_rows, n_cols, imn + 1)
                    # plt.title('Image '+str(imn)); plt.imshow(acts_sum); plt.colorbar(); plt.clim(0,1);
                    plt.title('Image ' + str(imn))
                    plt.imshow(x[imn], cmap='gray')
                    plt.imshow(resized, cmap='jet', alpha=0.25)
                    plt.colorbar()
                    plt.clim(0, 1)

                plt.suptitle('This are the activations for layer ' + str(model.layers[layers[0]].name), fontsize=14)
                # plt.show()

        plt_num = 0

        if type(activations) is list:
            # It's a list
            # It's a multi-layer output

            for activation in activations:

                plt_num = plt_num + 1

                if len(activation.shape) < 3:
                    # It's a dense layer output process as such

                    plt.figure(num=plt_num, figsize=(15 * 2, 8))

                    l, b = activation.shape

                    n_rows = round(np.sqrt(l))
                    n_cols = math.ceil(np.sqrt(l))

                    for imn in range(l):
                        actvtns = np.reshape(activation[imn], (1, b))

                        plt.subplot(n_rows, n_cols, imn + 1)
                        plt.title('Image ' + str(imn))
                        plt.imshow(actvtns)
                        plt.colorbar()
                        plt.clim(0, 1)

                    plt.suptitle('This are the activations for layer ' + str(model.layers[layers[plt_num - 1]].name),
                                 fontsize=14)
                    # plt.show()

                else:
                    # It's a convolutional layer output, process as such

                    plt.figure(num=plt_num, figsize=(15 * 2, 8))

                    l, w, h, b = activation.shape

                    n_rows = round(np.sqrt(l))
                    n_cols = math.ceil(np.sqrt(l))

                    for imn in range(l):
                        acts_sum = np.zeros((w, h))
                        for n in range(b):
                            acts_sum = acts_sum + activation[imn, :, :, n]

                        acts_sum *= 1.0 / acts_sum.max()
                        resized = resize(acts_sum, (x[imn].shape[0], x[imn].shape[1]))

                        # plt.figure(num=imn, figsize=(6,5));
                        plt.subplot(n_rows, n_cols, imn + 1)
                        plt.title('Image ' + str(imn))
                        plt.imshow(x[imn], cmap='gray')
                        plt.imshow(resized, cmap='jet', alpha=0.25)
                        plt.colorbar()
                        plt.clim(0, 1)

                    plt.suptitle('This are the activations for layer ' + str(model.layers[layers[plt_num - 1]].name),
                                 fontsize=14)
                    # plt.show()

        plt.show()

        return activations
