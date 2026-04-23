import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import numpy.typing as npt
from utils.utils import scale_session
import random

class StimulusMixin:

    @staticmethod
    def _load_images(
            image_dir: Path,
            flat: bool = False
    ) -> tuple[npt.NDArray[np.float32], list[str]]:
        """
        Loads PNG images from a directory, normalizes them to [0, 1],
        and optionally flattens them.
        """
        images = []
        image_names = []
        for image_path in sorted(image_dir.glob("*.png")):
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype(np.float32) / 255.0
            if flat:
                images.append(img.ravel())
            else:
                images.append(img)

            image_names.append(image_path.name)

        return np.array(images), image_names



    @staticmethod
    def _process_image_names(image_names_raw):
        ## ToDO: make this not hard coded
        src_cats = []
        dst_cats = []
        step_indeces = []
        stimuli_metadata = pd.DataFrame()
        src_cat_index = 0
        dst_cat_index = 2
        step_index_index = 5
        for i, img_name in enumerate(image_names_raw):
            label_split = img_name.split('_')
            src_cats.append(label_split[src_cat_index].lower())
            dst_cats.append(label_split[dst_cat_index].lower())
            step_indeces.append(int(label_split[step_index_index]))
        stimuli_metadata['src_cat'] = src_cats
        stimuli_metadata["dst_cat"] = dst_cats
        stimuli_metadata['step_index'] = step_indeces
        stimuli_metadata['pair_key'] = stimuli_metadata.apply(
            lambda row: "__".join(sorted([row['src_cat'], row['dst_cat']])),
            axis=1
        )
        return stimuli_metadata


    @staticmethod
    def _process_gabor(images: npt.NDArray, gabor_params: dict, output_dir: Path) -> pd.DataFrame:
        gabor_save_file = output_dir / "gabor_normalized_features_stimuli.npy"
        if Path.exists(gabor_save_file):
            print("Gabor Feature matrix already exists. Loading...")
            return pd.DataFrame(np.load(gabor_save_file))

        print("Starting Gabor Feature tranformations...")
        wavelengths = gabor_params["wavelengths"]
        orientations = gabor_params["orientations"]
        gamma = gabor_params["gamma"]
        grid_size = gabor_params["grid_size"]
        all_features = []
        gabor_img_dir =  output_dir / 'gabor_images'
        if not output_dir.exists(): output_dir.mkdir()
        if not gabor_img_dir.exists(): gabor_img_dir.mkdir()
        if len(images) <1:
            raise ValueError("Please provide at least one image to process....")
        for i, img in enumerate(images):
            print(f"Processing image {i + 1}/{len(images)}")
            image_vector = []
            for lambd in wavelengths:
                sigma = 0.5 * lambd
                for theta_deg in orientations:
                    theta = np.deg2rad(theta_deg)
                    ksize = int(lambd * 2) | 1
                    kernel_even = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                    kernel_odd = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, np.pi / 2, ktype=cv2.CV_32F)
                    res_even = cv2.filter2D(img, cv2.CV_32F, kernel_even)
                    res_odd = cv2.filter2D(img, cv2.CV_32F, kernel_odd)
                    magnitude = np.sqrt(res_even ** 2 + res_odd ** 2)
                    mag_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imwrite(gabor_img_dir / f"{i}_{lambd}_{theta_deg}.png", mag_vis)
                    pooled = cv2.resize(magnitude, grid_size, interpolation=cv2.INTER_AREA)
                    image_vector.append(pooled.flatten())
            all_features.append(np.concatenate(image_vector))
        feature_matrix = np.array(all_features)
        normalized_features = scale_session(feature_matrix)
        np.save(output_dir / 'gabor_normalized_features_stimuli.npy', normalized_features)
        return pd.DataFrame(normalized_features)

    @staticmethod
    def _process_distributed_gabor(images: npt.NDArray, gabor_params: dict, output_dir: Path, n_trials = 7) -> pd.DataFrame:
        gabor_save_file = output_dir / "gabor_distributed_normalized_features_stimuli.npy"
        if Path.exists(gabor_save_file):
            print("Gabor Feature matrix already exists. Loading...")
            return pd.DataFrame(np.load(gabor_save_file))

        wavelengths = gabor_params["wavelengths"]
        orientations = gabor_params["orientations"]
        gamma = gabor_params["gamma"]
        receptive_field_sizes = gabor_params["receptive_field_sizes"]
        n_neurons = gabor_params["n_neurons"]
        fano_factor = 1.2
        sensor_noise_std = 0.05

        neuron_param_dict = {}
        for i in range(n_neurons):
            neuron_param_dict[i] = {}
            neuron_param_dict[i]["orientation"] = random.choice(orientations)
            neuron_param_dict[i]["wavelength"] = random.choice(wavelengths)
            neuron_param_dict[i]["gamma"] = gamma
            receptive_field_size = random.choice(receptive_field_sizes)
            img_shape = images[0].shape
            while True:
                receptive_field_location = (np.random.randint(0, img_shape[0]), np.random.randint(0, img_shape[1]))
                x1 = receptive_field_location[0] - receptive_field_size // 2
                y1 = receptive_field_location[1] - receptive_field_size // 2
                x2 = receptive_field_location[0] + receptive_field_size // 2
                y2 = receptive_field_location[1] + receptive_field_size // 2
                if 0 < min([x1,x2,y1,y2]) < max(img_shape):
                    break
            receptive_field = [[x1,y1], [x2,y2]]
            neuron_param_dict[i]["receptive_field"] = receptive_field

        final_feature_matrix = np.zeros((len(images) * n_trials, n_neurons))
        for i, img in enumerate(images):
            img_num_after_noise = i * n_trials
            if i % 10 == 0:  # Simple progress tracker
                print(f"Processing image {i + 1}/{len(images)}")

            for j, neuron_params in neuron_param_dict.items():
                orientation = neuron_params["orientation"]
                wavelength = neuron_params["wavelength"]
                gamma = neuron_params["gamma"]
                (x1, y1), (x2, y2) = neuron_params["receptive_field"]
                img_crop = img[y1:y2, x1:x2]

                theta = np.deg2rad(orientation)
                sigma = 0.5 * wavelength
                ksize = int(wavelength * 2) | 1
                kernel_even = cv2.getGaborKernel((ksize, ksize), sigma, theta, wavelength, gamma, 0, ktype=cv2.CV_32F)
                kernel_odd = cv2.getGaborKernel((ksize, ksize), sigma, theta, wavelength, gamma, np.pi / 2,
                                                ktype=cv2.CV_32F)
                res_even = cv2.filter2D(img_crop, cv2.CV_32F, kernel_even)
                res_odd = cv2.filter2D(img_crop, cv2.CV_32F, kernel_odd)

                magnitude = np.sqrt(res_even ** 2 + res_odd ** 2)

                base_activation = np.mean(magnitude)

                for trial in range(n_trials):
                    # 1. Apply Poisson-like variability
                    # We scale by fano_factor to control "cleanliness"
                    trial_activation = np.random.poisson(base_activation * 100) / 100.0

                    # 2. Add Gaussian "Instrument" noise
                    noise = np.random.normal(0, sensor_noise_std)

                    # 3. Final value (rectified to ensure no negative firing)
                    final_feature_matrix[img_num_after_noise+trial, j] = max(0, trial_activation + noise)


            # 3. Scale and Save
        print("Scaling and saving features...")
        normalized_features = scale_session(final_feature_matrix)
        np.save(gabor_save_file, normalized_features)

        return pd.DataFrame(normalized_features)

        teset=1


        ## for each neuron, take only the receptive field
        # calculate the gabor values for that ne