
import numpy.typing as npt
from src.utils.utils import scale_session, format_seconds
from concurrent.futures import ProcessPoolExecutor

import time
import random
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import concurrent.futures


def _process_single_image(img_idx, img, neuron_param_dict, n_trials, fano_factor, sensor_noise_std):
    n_neurons = len(neuron_param_dict)
    img_features = np.zeros((n_trials, n_neurons))

    for j, neuron_params in neuron_param_dict.items():
        (x1, y1), (x2, y2) = neuron_params["receptive_field"]
        img_crop = img[y1:y2, x1:x2]

        if img_crop.size == 0:
            raise AssertionError("Cropped image does not contain pixels")

        res_even = cv2.filter2D(img_crop, cv2.CV_32F, neuron_params["kernel_even"])
        res_odd = cv2.filter2D(img_crop, cv2.CV_32F, neuron_params["kernel_odd"])

        magnitude = np.sqrt(res_even ** 2 + res_odd ** 2)
        base_activation = np.mean(magnitude)

        mu = base_activation * 100
        if mu > 0:
            poisson_counts = np.random.poisson(mu / fano_factor, size=n_trials)
            trial_activations = poisson_counts.astype(float) * fano_factor
        else:
            trial_activations = np.zeros(n_trials, dtype=float)

        trial_activations /= 100.0
        noise = np.random.normal(0, sensor_noise_std, size=n_trials)

        img_features[:, j] = np.maximum(0, trial_activations + noise)

    return img_idx, img_features

def create_distributed_gabor(images: np.ndarray, gabor_params: dict, output_dir: Path) -> pd.DataFrame:
    gabor_save_file = output_dir / "GaborNetCalculatedCache.npy"
    if not gabor_params['recalculate_gabornet']:
        if gabor_save_file.exists():
            print('GaborNet Feature Matrix found. Loading...')
            return pd.DataFrame(np.load(gabor_save_file))

    # Extract parameters
    wavelengths = gabor_params["wavelengths"]
    gamma = gabor_params["gamma"]
    receptive_field_sizes = gabor_params["receptive_field_sizes"]
    n_neurons = gabor_params["n_neurons"]
    n_trials = gabor_params["n_trials"]
    fano_factor = gabor_params["fano_factor"]
    sensor_noise_std = gabor_params["sensor_noise_std"]
    orientation_dict = gabor_params["orientation_dict"]
    orientations = [int(key) for key in orientation_dict.keys()]
    orientation_probs = list(orientation_dict.values())

    # Build neuron parameters AND precompute Gabor kernels
    print("Generating neurons and precomputing Gabor kernels...")
    neuron_param_dict = {}
    for i in range(n_neurons):
        neuron_param_dict[i] = {}
        orientation = np.random.choice(orientations, 1, p=orientation_probs)[0]
        wavelength = random.choice(wavelengths)

        neuron_param_dict[i]["orientation"] = orientation
        neuron_param_dict[i]["wavelength"] = wavelength
        neuron_param_dict[i]["gamma"] = gamma

        # Receptive field logic
        while True:
            receptive_field_size = random.choice(receptive_field_sizes) * 4
            img_shape = images[0].shape
            if receptive_field_size < min(img_shape) / 2:
                break
        while True:
            receptive_field_location = (np.random.randint(0, img_shape[0]), np.random.randint(0, img_shape[1]))
            x1 = receptive_field_location[1] - receptive_field_size // 2
            x2 = x1 + receptive_field_size
            y1 = receptive_field_location[0] - receptive_field_size // 2
            y2 = y1 + receptive_field_size
            if 0 <= x1 < x2 <= img_shape[1] and 0 <= y1 < y2 <= img_shape[0]:
                break

        neuron_param_dict[i]["receptive_field"] = [[x1, y1], [x2, y2]]

        # Precompute kernels here so we only do it once per neuron total!
        theta = np.deg2rad(orientation)
        sigma = 0.5 * wavelength
        ksize = int(wavelength * 2) | 1
        neuron_param_dict[i]["kernel_even"] = cv2.getGaborKernel((ksize, ksize), sigma, theta, wavelength, gamma, 0,
                                                                 ktype=cv2.CV_32F)
        neuron_param_dict[i]["kernel_odd"] = cv2.getGaborKernel((ksize, ksize), sigma, theta, wavelength, gamma,
                                                                np.pi / 2, ktype=cv2.CV_32F)

    # 3. Parallelize across images using Multiprocessing
    final_feature_matrix = np.zeros((len(images) * n_trials, n_neurons))
    start_time = time.time()

    print(f"Processing {len(images)} images across parallel workers...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks to the pool
        futures = [
            executor.submit(_process_single_image, i, img, neuron_param_dict, n_trials, fano_factor, sensor_noise_std)
            for i, img in enumerate(images)
        ]

        # Process results as they complete to keep the live ETA working
        for completed_count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            img_idx, img_features = future.result()

            # Map the local chunk back into the global matrix in the correct spot
            start_row = img_idx * n_trials
            end_row = start_row + n_trials
            final_feature_matrix[start_row:end_row, :] = img_features

            # ETA Calculations
            elapsed_seconds = time.time() - start_time
            avg_time_per_img = elapsed_seconds / completed_count
            images_remaining = len(images) - completed_count
            eta_seconds = avg_time_per_img * images_remaining

            print(f"Processed image {completed_count}/{len(images)} | "
                  f"Time Taken: {format_seconds(elapsed_seconds)} | "
                  f"ETA: {format_seconds(eta_seconds)}")

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