import cv2
from sklearn.decomposition import PCA
import os
from pathlib import Path
from os.path import join
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.io import loadmat

class Stimuli:
    """"
    Class that handles t
    """
    def __init__(self):
        self.images = []
        self.images_flat = []
        self.images_gabor = []
        self.image_names_raw = []
        self.pca_dict = {}
        self.data_dir = Path()
        self.stimuli_metadata = pd.DataFrame()

    def set_data_dir(self, data_dir):
        self.data_dir = Path(data_dir)

    def load_images(self ):
        image_dir = Path(join(self.data_dir, 'stimuli'))
        paths = sorted(image_dir.iterdir())

        for image_path in paths:
            if not str(image_path).endswith(".png"): continue
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            self.images.append(img)
            self.images_flat.append(img.flatten())
            self.image_names_raw.append(image_path.name)

        self._proces_image_names()

    def _proces_image_names(self):
        ## ToDO: make this not hard coded
        orig_cats = []
        dst_cats = []
        step_indeces = []
        stimuli_metadata = pd.DataFrame()
        orig_cat_index = 0
        dst_cat_index = 2
        step_index_index = 5

        for i, img_name in enumerate(self.image_names_raw):
            label_split =  img_name.split('_')
            orig_cats.append(label_split[orig_cat_index].lower())
            dst_cats.append(label_split[dst_cat_index].lower())
            step_indeces.append(label_split[step_index_index].lower())
        stimuli_metadata['orig_cat'] = orig_cats
        stimuli_metadata["dst_cat"] = dst_cats
        stimuli_metadata['step_index'] = step_indeces
        stimuli_metadata['pair_key'] = stimuli_metadata['orig_cat'] + '__' + stimuli_metadata["dst_cat"]
        stimuli_metadata['full_name'] = stimuli_metadata['pair_key'] + '__' + stimuli_metadata['step_index'].astype(str)
        max_index = max(stimuli_metadata['step_index'].astype(int))
        stim_types = ['full' if ind == "00" or ind == str(max_index) else 'morph' for ind in stimuli_metadata['step_index'].values]
        stimuli_metadata['stim_type'] = stim_types

        self.stimuli_metadata = stimuli_metadata







    def process_images(self, n_components_pixel, n_components_gabor, gabor_params):


        self._process_gabor(gabor_params)
        self._process_pca(n_components_pixel, n_components_gabor)
        return  self.pca_dict




    def _process_gabor(self, gabor_params):

        print("Starting Gabor Feature tranformations...")
        wavelengths = gabor_params["wavelengths"]
        orientations = gabor_params["orientations"]
        gamma = gabor_params["gamma"]
        grid_size = gabor_params["grid_size"]
        all_features = []

        output_dir = join(self.data_dir, "output")
        gabor_img_dir =  join(output_dir, "gabor_images")
        if not os.path.exists(output_dir): os.makedirs(join(output_dir))
        if not os.path.exists(gabor_img_dir): os.makedirs(gabor_img_dir)

        for i, img in enumerate(self.images):
            # FIX: Ensure image is 2D (512, 512) for filter2D to work

            print(f"Processing image {i + 1}/{len(self.images)}")
            image_vector = []

            for lambd in wavelengths:
                sigma = 0.5 * lambd

                for theta_deg in orientations:
                    theta = np.deg2rad(theta_deg)
                    ksize = int(lambd * 2) | 1

                    # 1. Generate Kernels
                    kernel_even = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                    kernel_odd = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, np.pi / 2,
                                                    ktype=cv2.CV_32F)

                    # 2. Convolve
                    res_even = cv2.filter2D(img, cv2.CV_32F, kernel_even)
                    res_odd = cv2.filter2D(img, cv2.CV_32F, kernel_odd)

                    # 3. Compute Magnitude Response
                    magnitude = np.sqrt(res_even ** 2 + res_odd ** 2)

                    # VISUALIZATION: Normalize for saving to disk (0-255)
                    mag_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imwrite(join(gabor_img_dir, f"{i}_{lambd}_{theta_deg}.png"), mag_vis)

                    # 4. Spatial Pooling (Downsampling 512x512 -> 8x8)
                    pooled = cv2.resize(magnitude, grid_size, interpolation=cv2.INTER_AREA)

                    image_vector.append(pooled.flatten())

            all_features.append(np.concatenate(image_vector))

        feature_matrix = np.array(all_features)

        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)

        np.save(join(output_dir, 'gabor_normalized_features_stimuli.npy'), normalized_features)

        self.images_gabor = normalized_features

    def _process_pca(self, n_components_pixel, n_components_gabor):
        pca_pixel_model = PCA(n_components=n_components_pixel)
        pca_gabor_model = PCA(n_components=n_components_gabor)

        self.pca_dict["pixel"] = pca_pixel_model.fit_transform(self.images_flat), None, self.stimuli_metadata
        self.pca_dict["gabor"] = pca_gabor_model.fit_transform(self.images_gabor), None, self.stimuli_metadata
