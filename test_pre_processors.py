from utils import FileManager
from pre_processors import ImagePreProcessor, ChestXRayPreProcessor
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == "__main__":

    extract_hog = False
    compute_pca = False
    get_image_data = False
    segment_lungs = True
    pre_process_cnn = True

    # -------------------------------------- Hog Features --------------------------------------

    if extract_hog:
        pr = ImagePreProcessor()
        x = np.random.random((100, 10, 10))
        hog_array = pr.extract_hog(x)

    # ------------------------------------------- PCA -------------------------------------------

    if compute_pca:
        pr = ImagePreProcessor()
        x = np.random.random((100, 10))
        n_pca_components = 10
        p_components = pr.pca(x, n_pca_components=n_pca_components)

    # -------------------------------------- Segment lungs --------------------------------------

    if segment_lungs:
        pr = ChestXRayPreProcessor()
        training_path = 'covid19_repo/data/x_rays/covid'
        model_path = 'covid19_repo/u_net_model/Unet_model.json'
        weights_path = 'covid19_repo/u_net_model/Unet_model.h5'
        batch_size = 16
        threshold = 0.1
        verbose = 1

        fm = FileManager()
        x = np.load('data/x_512.npy')
        segmented_lungs = pr.segment_lungs(x,
                                           model_path=model_path,
                                           weights_path=weights_path,
                                           batch_size=batch_size,
                                           threshold=threshold,
                                           verbose=verbose)
        plt.figure()
        plt.imshow(segmented_lungs[np.random.randint(0, len(segmented_lungs))])

    if pre_process_cnn:
        pr = ImagePreProcessor()
        x = np.load('data/x_512.npy')[0:10]
        x = pr.pre_process_built_cnn(x, type_cnn='resnet50')
        plt.figure()
        plt.imshow(x[np.random.randint(0, len(x))])
