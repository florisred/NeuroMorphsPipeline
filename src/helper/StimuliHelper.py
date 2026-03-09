from os.path import join
import numpy as np
from sklearn.preprocessing import StandardScaler
import cv2
import os
import pandas as pd
from pathlib import Path

class StimuliHelper:
    def _create_data_dict(self, images_flat, gabor_bank, stimuli_metadata):
        keep_mask = ~stimuli_metadata['morph_name'].duplicated(keep='first')
        stimuli_metadata = stimuli_metadata[keep_mask]
        gabor_bank = gabor_bank[keep_mask]
        images_flat = images_flat[keep_mask]
        if self.data_dict is None: self.data_dict = {}

        self.data_dict["pixel"] = {
            "data": images_flat,
            "labels": stimuli_metadata,
        }
        self.data_dict["gabor"] = {
            "data": gabor_bank,
            "labels": stimuli_metadata,
        }

    def _load_images(self):

        image_dir = Path(join(self.data_dir, 'stimuli'))
        paths = sorted(image_dir.iterdir())
        images, images_flat_df, image_names_raw = self._read_files(paths)
        stimuli_metadata = self._process_image_names(image_names_raw)
        return images, images_flat_df, stimuli_metadata

    @staticmethod
    def _read_files(paths):
        images_flat = []
        image_names_raw = []
        images = []
        for image_path in paths:
            if not str(image_path).endswith(".png"): continue
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            images.append(img)
            images_flat.append(img.flatten())
            image_names_raw.append(image_path.name)
        return images, pd.DataFrame(images_flat), image_names_raw

    @staticmethod
    def _process_image_names(image_names_raw):
        ## ToDO: make this not hard coded
        src_cats = []
        dst_cats = []
        step_indeces = []
        stimuli_metadata = pd.DataFrame()
        orig_cat_index = 0
        dst_cat_index = 2
        step_index_index = 5
        for i, img_name in enumerate(image_names_raw):
            label_split = img_name.split('_')
            src_cats.append(label_split[orig_cat_index].lower())
            dst_cats.append(label_split[dst_cat_index].lower())
            step_indeces.append(int(label_split[step_index_index]))
        stimuli_metadata['src_cat'] = src_cats
        stimuli_metadata["dst_cat"] = dst_cats
        stimuli_metadata['step_index'] = step_indeces
        range_index = (min(stimuli_metadata['step_index']), max(stimuli_metadata['step_index']))
        stim_types = ['anchor' if ind in range_index else 'morph' for ind in
                      stimuli_metadata['step_index'].values]
        stimuli_metadata['norm_step'] = stimuli_metadata['step_index'] / range_index[1]
        stimuli_metadata['pair_key'] = stimuli_metadata.apply(
            lambda row: "__".join(sorted([row['src_cat'], row['dst_cat']])),
            axis=1
        )
        stimuli_metadata['stim_type'] = stim_types
        stimuli_metadata.loc[stimuli_metadata['stim_type'] == 'anchor', 'pair_key'] = np.nan
        stimuli_metadata['morph_name'] = stimuli_metadata.apply(
            lambda row: row['src_cat'] + "_" +
                        str(round(1 - row['norm_step'], 2)) + "_" + row['dst_cat'] + "_" +
                        str(round(row['norm_step'], 2)),
            axis=1
        )
        stimuli_metadata.loc[stimuli_metadata['norm_step'] == 0.0, 'morph_name'] = stimuli_metadata['src_cat']
        stimuli_metadata.loc[stimuli_metadata['norm_step'] == 1.0, 'morph_name'] = stimuli_metadata['dst_cat']

        return stimuli_metadata


    def _process_gabor(self, images, gabor_params):
        if Path.exists(Path(join(self.data_dir, "output", "gabor_normalized_features_stimuli.npy"))):
            print("Gabor Feature matrix already exists. Loading...")
            return pd.DataFrame(np.load(join(self.data_dir, "output", "gabor_normalized_features_stimuli.npy")))
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

        for i, img in enumerate(images):

            print(f"Processing image {i + 1}/{len(images)}")
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
        return pd.DataFrame(normalized_features)