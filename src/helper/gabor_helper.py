import random
from pathlib import Path

import cv2
import numpy.typing as npt
import pandas as pd
import numpy as np
from src.utils.utils import scale_session





def create_distributed_gabor(images: npt.NDArray, gabor_params: dict, output_dir: Path, n_trials = 1, recalculate=False) -> pd.DataFrame:
        gabor_save_file = output_dir / "GaborNetCalculatedCache.npy"
        if not recalculate:
            if Path.exists(gabor_save_file):
                print('GaborNet Feature Matrix found. Loading...')
                return pd.DataFrame(np.load(gabor_save_file))
        wavelengths = gabor_params["wavelengths"]
        gamma = gabor_params["gamma"]
        receptive_field_sizes = gabor_params["receptive_field_sizes"]
        n_neurons = gabor_params["n_neurons"]
        fano_factor = 20
        sensor_noise_std = 2
        neuron_param_dict = {}
        orientation_dict = gabor_params["orientation_dict"]
        orientations = [int(key) for key in orientation_dict.keys()]
        orientation_probs = list(orientation_dict.values())
        for i in range(n_neurons):
            neuron_param_dict[i] = {}
            neuron_param_dict[i]["orientation"] = np.random.choice(orientations, 1, p=orientation_probs)[0]
            neuron_param_dict[i]["wavelength"] = random.choice(wavelengths)
            neuron_param_dict[i]["gamma"] = gamma
            while True:
                receptive_field_size = random.choice(receptive_field_sizes) * 4
                img_shape = images[0].shape
                if receptive_field_size < min(img_shape) / 2: break
            while True:
                receptive_field_location = (np.random.randint(0, img_shape[0]), np.random.randint(0, img_shape[1]))
                x1 = receptive_field_location[1] - receptive_field_size // 2
                x2 = x1+receptive_field_size
                y1 = receptive_field_location[0] - receptive_field_size // 2
                y2 = y1 + receptive_field_size
                x_possible = 0 <min(x1,x2) < max(x1,x2) < img_shape[1]
                y_possible = 0 < min(y1,y2) < max(y1,y2) < img_shape[0]
                if x_possible and y_possible:
                    break
            receptive_field = [[x1,y1], [x2,y2]]
            neuron_param_dict[i]["receptive_field"] = receptive_field
        final_feature_matrix = np.zeros((len(images) * n_trials, n_neurons))
        for i, img in enumerate(images):
            img_num_after_noise = i * n_trials
            if i % 10 == 0:
                print(f"Processing image {i + 1}/{len(images)}")
            for j, neuron_params in neuron_param_dict.items():
                orientation = neuron_params["orientation"]
                wavelength = neuron_params["wavelength"]
                gamma = neuron_params["gamma"]
                (x1, y1), (x2, y2) = neuron_params["receptive_field"]
                img_crop = img[y1:y2, x1:x2]
                if min(img_crop.shape) == 0:
                    raise AssertionError("Cropped image does not contain pixels")

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
                    mu = base_activation * 100  # Your scaling factor
                    if mu > 0:
                        scaled_mu = mu / fano_factor
                        trial_activation = np.random.poisson(scaled_mu) * fano_factor
                    else:
                        trial_activation = 0.0
                    trial_activation /= 100.0
                    noise = np.random.normal(0, sensor_noise_std)
                    final_feature_matrix[img_num_after_noise + trial, j] = max(0, trial_activation + noise)
                # for trial in range(n_trials):
                #     mu = base_activation * 100  # Your scaling factor
                #
                #     # --- NOISE DISABLED HERE ---
                #     # Instead of sampling randomly, just assign the clean mean directly
                #     trial_activation = mu / 100.0
                #
                #     final_feature_matrix[img_num_after_noise + trial, j] = max(0, trial_activation)
        print("Scaling and saving features...")
        normalized_features = scale_session(final_feature_matrix)
        np.save(gabor_save_file, normalized_features)

        return pd.DataFrame(normalized_features)


def process_gabor(images: npt.NDArray, gabor_params: dict, output_dir: Path) -> pd.DataFrame:

    """
    The main function that creates the gabor filter bank used in the analyses.
    Takes the images in their normal shape, a dictionary of the gabor params, and an output dir, where it should save
    both the processed images and a normalized_features.npy file. This file can be then loaded again to save time
    """
    # init path of save file
    gabor_save_file = output_dir / "gabor_normalized_features_stimuli.npy"
    if Path.exists(gabor_save_file):
        print("Gabor Feature matrix already exists. Loading...")
        return pd.DataFrame(np.load(gabor_save_file))

    print("Starting Gabor Feature tranformations...")
    # loads the gabor parameters
    wavelengths = gabor_params["wavelengths"]
    orientation_dict = gabor_params["orientation_dict"]
    orientations = [int(key) for key in orientation_dict.keys()]
    gamma = gabor_params["gamma"]
    grid_size = gabor_params["grid_size"]
    all_features = []

    #creates or loads the gabor images folder
    gabor_img_dir =  output_dir / 'gabor_images'
    if not output_dir.exists(): output_dir.mkdir()
    if not gabor_img_dir.exists(): gabor_img_dir.mkdir()

    if len(images) <1:
        raise ValueError("Please provide at least one image to process....")

    # applies the gabor filter bank on a per-image basis
    for i, img in enumerate(images):
        print(f"Processing image {i + 1}/{len(images)}")
        image_vector = []

        # for each wavelength and orientation combination, it creates a separate 'gabor-filtered image'
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