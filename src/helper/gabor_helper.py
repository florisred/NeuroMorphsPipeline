import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random
import time
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom
from src.utils.utils import format_seconds, scale_session
import random


def _process_single_image(
    img_idx, img, neuron_param_dict, n_trials, fano_factor, sensor_noise_std
):
    n_neurons = len(neuron_param_dict)
    img_features = np.zeros((n_trials, n_neurons))

    for j, neuron_params in neuron_param_dict.items():
        (x1, y1), (x2, y2) = neuron_params["receptive_field"]
        img_crop = img[y1:y2, x1:x2]

        if img_crop.size == 0:
            raise AssertionError("Cropped image does not contain pixels")

        res_even = cv2.filter2D(
            img_crop, cv2.CV_32F, neuron_params["kernel_even"]
        )
        res_odd = cv2.filter2D(img_crop, cv2.CV_32F, neuron_params["kernel_odd"])

        magnitude = np.sqrt(res_even**2 + res_odd**2)
        base_activation = np.mean(magnitude)

        mu = base_activation * 10
        if mu > 0:
            poisson_counts = np.random.poisson(mu / fano_factor, size=n_trials)
            trial_activations = poisson_counts.astype(float) * fano_factor
        else:
            trial_activations = np.zeros(n_trials, dtype=float)
        noise = np.random.normal(0, sensor_noise_std, size=n_trials)
        img_features[:, j] = np.maximum(0, trial_activations + noise)

    return img_idx, img_features


def create_distributed_gabor(
    images: np.ndarray, gabor_params: dict, output_dir: Path
) -> pd.DataFrame:
    gabor_save_file = output_dir / "GaborNetCalculatedCache.npy"
    if not gabor_params["recalculate_gabornet"]:
        if gabor_save_file.exists():
            print("GaborNet Feature Matrix found. Loading...")
            return pd.DataFrame(np.load(gabor_save_file))

    # Extract parameters
    gamma = gabor_params["gamma"]
    receptive_field_sizes = gabor_params["receptive_field_sizes"]
    n_neurons = gabor_params["n_neurons"]
    n_trials = gabor_params["n_trials"]
    fano_factor = gabor_params["fano_factor"]
    sensor_noise_std = gabor_params["sensor_noise_std"]
    orientation_dict = gabor_params["orientation_dict"]
    orientations = [int(key) for key in orientation_dict.keys()]
    orientation_probs = list(orientation_dict.values())

    print("Generating neurons and precomputing Gabor kernels...")
    neuron_param_dict = {}
    img_shape = images[0].shape

    for i in range(n_neurons):
        neuron_param_dict[i] = {}

        orientation = np.random.choice(orientations, 1, p=orientation_probs)[0]
        neuron_param_dict[i]["orientation"] = orientation
        neuron_param_dict[i]["gamma"] = gamma

        receptive_field_size = random.choice(receptive_field_sizes)

        cycles = np.random.uniform(2.3, 3.5)
        wavelength = receptive_field_size / cycles
        neuron_param_dict[i]["wavelength"] = wavelength

        ksize = int(wavelength * 2) | 1
        if ksize > receptive_field_size:
            ksize = (
                receptive_field_size
                if (receptive_field_size % 2 == 1)
                else (receptive_field_size - 1)
            )

        center_y, center_x = img_shape[0] // 2, img_shape[1] // 2

        sigma_y, sigma_x = img_shape[0] // 4, img_shape[1] // 4

        while True:
            rf_center_y = int(np.random.normal(center_y, sigma_y))
            rf_center_x = int(np.random.normal(center_x, sigma_x))

            x1 = rf_center_x - receptive_field_size // 2
            x2 = x1 + receptive_field_size
            y1 = rf_center_y - receptive_field_size // 2
            y2 = y1 + receptive_field_size
            if 0 <= x1 and x2 <= img_shape[1] and 0 <= y1 and y2 <= img_shape[0]:
                break
            else:
                print(
                    f"crop out of bounds with receptive field size {receptive_field_size}, retrying with a smaller receptive field size..."
                )
                if receptive_field_size > 10:
                    receptive_field_size = receptive_field_size - 5

        neuron_param_dict[i]["receptive_field"] = [[x1, y1], [x2, y2]]

        theta = np.deg2rad(orientation)
        sigma = 0.5 * wavelength
        neuron_param_dict[i]["kernel_even"] = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, wavelength, gamma, 0, ktype=cv2.CV_32F
        )
        neuron_param_dict[i]["kernel_odd"] = cv2.getGaborKernel(
            (ksize, ksize),
            sigma,
            theta,
            wavelength,
            gamma,
            np.pi / 2,
            ktype=cv2.CV_32F,
        )
    final_feature_matrix = np.zeros((len(images) * n_trials, n_neurons))
    start_time = time.time()

    print(f"Processing {len(images)} images across parallel workers...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _process_single_image,
                i,
                img,
                neuron_param_dict,
                n_trials,
                fano_factor,
                sensor_noise_std,
            )
            for i, img in enumerate(images)
        ]

        for completed_count, future in enumerate(
            concurrent.futures.as_completed(futures), 1
        ):
            img_idx, img_features = future.result()

            start_row = img_idx * n_trials
            end_row = start_row + n_trials
            final_feature_matrix[start_row:end_row, :] = img_features

            elapsed_seconds = time.time() - start_time
            avg_time_per_img = elapsed_seconds / completed_count
            images_remaining = len(images) - completed_count
            eta_seconds = avg_time_per_img * images_remaining

            print(
                f"Processed image {completed_count}/{len(images)} | "
                f"Time Taken: {format_seconds(elapsed_seconds)} | "
                f"ETA: {format_seconds(eta_seconds)}"
            )

    print("Scaling and saving features...")
    normalized_features = scale_session(final_feature_matrix)
    np.save(gabor_save_file, normalized_features)

    return pd.DataFrame(normalized_features)


def process_gabor(
    images: npt.NDArray, gabor_params: dict, output_dir: Path
) -> pd.DataFrame:
    """The main function that creates the gabor filter bank used in the analyses.

    Takes the images in their normal shape, a dictionary of the gabor params,
    and an output dir, where it should save both the processed images and a
    normalized_features.npy file. This file can be then loaded again to save
    time
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

    # creates or loads the gabor images folder
    gabor_img_dir = output_dir / "gabor_images"
    if not output_dir.exists():
        output_dir.mkdir()
    if not gabor_img_dir.exists():
        gabor_img_dir.mkdir()

    if len(images) < 1:
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
                kernel_even = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F
                )
                kernel_odd = cv2.getGaborKernel(
                    (ksize, ksize),
                    sigma,
                    theta,
                    lambd,
                    gamma,
                    np.pi / 2,
                    ktype=cv2.CV_32F,
                )
                res_even = cv2.filter2D(img, cv2.CV_32F, kernel_even)
                res_odd = cv2.filter2D(img, cv2.CV_32F, kernel_odd)
                magnitude = np.sqrt(res_even**2 + res_odd**2)
                mag_vis = cv2.normalize(
                    magnitude, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                cv2.imwrite(
                    gabor_img_dir / f"{i}_{lambd}_{theta_deg}.png", mag_vis
                )
                pooled = cv2.resize(
                    magnitude, grid_size, interpolation=cv2.INTER_AREA
                )
                image_vector.append(pooled.flatten())
        all_features.append(np.concatenate(image_vector))
    feature_matrix = np.array(all_features)
    normalized_features = scale_session(feature_matrix)
    np.save(
        output_dir / "gabor_normalized_features_stimuli.npy",
        normalized_features,
    )
    return pd.DataFrame(normalized_features)


def _build_spatial_normalization_weights(neuron_param_dict, spatial_sigma):
    """
    Precomputes a row-normalized Gaussian weight matrix based on the physical
    distance between neuron receptive field centers.
    """
    n = len(neuron_param_dict)
    centers = np.zeros((n, 2))

    for i in range(n):
        (x1, y1), (x2, y2) = neuron_param_dict[i]["receptive_field"]
        centers[i] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    # Calculate all pairwise Euclidean distances between RF centers
    dist_matrix = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)

    # Convert distances to Gaussian weights
    weight_matrix = np.exp(-(dist_matrix ** 2) / (2 * (spatial_sigma ** 2)))

    # Row-normalize so every neuron calculates a weighted mean of its neighbors
    row_sums = weight_matrix.sum(axis=1, keepdims=True)
    weight_matrix = weight_matrix / row_sums

    return weight_matrix.astype(np.float32)


def _process_single_image_divnorm(
        img_idx, img, neuron_param_dict, weight_matrix, n_trials, fano_factor, sensor_noise_std, dn_params
):
    """Processes a single image, applying Gabor filters followed by Spatially-Weighted
    Divisive Normalization across the neural population before generating noisy trials.
    """
    n_neurons = len(neuron_param_dict)
    img_features = np.zeros((n_trials, n_neurons))
    raw_activations = np.zeros(n_neurons)

    # --- Pass 1: Compute raw linear responses ---
    for j, neuron_params in neuron_param_dict.items():
        (x1, y1), (x2, y2) = neuron_params["receptive_field"]
        neuron_type = neuron_params["type"]
        img_crop = img[y1:y2, x1:x2]

        if img_crop.size == 0:
            raise AssertionError("Cropped image does not contain pixels")

        res_even = cv2.filter2D(img_crop, cv2.CV_32F, neuron_params["kernel_even"])
        if neuron_type == 'complex':
            res_odd = cv2.filter2D(img_crop, cv2.CV_32F, neuron_params["kernel_odd"])
            magnitude = np.sqrt(res_even ** 2 + res_odd ** 2)
        elif neuron_type == 'simple':
            magnitude = np.maximum(0, res_even)
        else:
            raise AssertionError("Unknown neuron type")

        raw_activations[j] = np.mean(magnitude)

    # --- Pass 2: Apply Localized Divisive Normalization ---
    sigma_base = dn_params["sigma"]
    n = dn_params["n"]
    gain = dn_params["gain"]

    if dn_params.get("dynamic_sigma", False):
        sigma = sigma_base * np.mean(raw_activations)
    else:
        sigma = sigma_base

    activated_powered = raw_activations ** n
    normalization_pool = weight_matrix @ activated_powered

    denom = (sigma ** n) + normalization_pool
    normalized_activations = activated_powered / denom if np.all(denom > 0) else np.zeros_like(activated_powered)
    spont_floor = dn_params.get("spontaneous_rate", 0.0)

    for j in range(n_neurons):
        mu = (normalized_activations[j] * gain) + spont_floor

        if mu > 0:
            poisson_counts = np.random.poisson(mu / fano_factor, size=n_trials)
            trial_activations = poisson_counts.astype(float) * fano_factor
        else:
            trial_activations = np.zeros(n_trials, dtype=float)

        noise = np.random.normal(0, sensor_noise_std, size=n_trials)
        img_features[:, j] = np.maximum(0, trial_activations + noise)

    return img_idx, img_features


def create_retinodivnorm_gabornet(
        images: np.ndarray, gabor_params: dict, output_dir: Path
) -> pd.DataFrame:
    gabor_save_file = output_dir / "RetinoDivNormGaborNetCalculatedCache.npy"
    if not gabor_params.get("recalculate_gabornet", True):
        if gabor_save_file.exists():
            print("GaborNet Feature Matrix found. Loading...")
            return pd.DataFrame(np.load(gabor_save_file))

    # Extract parameters
    gamma = gabor_params["gamma"]
    receptive_field_sizes = gabor_params["receptive_field_sizes"]
    receptive_field_sizes = [int(min(rf, 250)) for rf in receptive_field_sizes]
    receptive_field_sizes = [int(max(rf, 8)) for rf in receptive_field_sizes]
    print(f"Mean receptive_field_sizes: {np.mean(receptive_field_sizes)}")

    n_neurons = gabor_params["n_neurons"]
    n_trials = gabor_params["n_trials"]
    fano_factor = gabor_params["fano_factor"]
    sensor_noise_std = gabor_params["sensor_noise_std"]
    orientation_dict = gabor_params["orientation_dict"]
    orientations = [int(key) for key in orientation_dict.keys()]
    orientation_probs = list(orientation_dict.values())

    # Default Divisive Normalization parameters updated based on biology
    dn_params = gabor_params.get(
        "dn_params",
        {
            "sigma": 0.3,
            "n": 2.0,
            "gain": 20.0,
            "spatial_sigma": 150,
            "dynamic_sigma": False,
            "spontaneous_rate":1,
            'probability_simple': 0.7
        }
    )

    print("Generating neurons and precomputing Gabor kernels...")
    neuron_param_dict = {}
    img_shape = images[0].shape

    center_y, center_x = img_shape[0] // 2, img_shape[1] // 2
    sigma_y, sigma_x = img_shape[0] // 4, img_shape[1] // 4

    for i in range(n_neurons):
        neuron_param_dict[i] = {}

        orientation = np.random.choice(orientations, 1, p=orientation_probs)[0]
        neuron_param_dict[i]["orientation"] = orientation
        neuron_param_dict[i]["gamma"] = gamma
        neuron_param_dict[i]['type'] = 'simple' if random.uniform(0, 1) < dn_params['probability_simple'] else 'complex'

        receptive_field_size = random.choice(receptive_field_sizes)

        cycles = np.random.uniform(1.0, 3.0)
        wavelength = receptive_field_size / cycles
        neuron_param_dict[i]["wavelength"] = wavelength

        ksize = int(wavelength * 2) | 1
        if ksize > receptive_field_size:
            ksize = receptive_field_size if (receptive_field_size % 2 == 1) else (receptive_field_size - 1)

        half = receptive_field_size // 2
        min_y, max_y = half, img_shape[0] - half
        min_x, max_x = half, img_shape[1] - half

        rf_center_y = int(np.clip(np.random.normal(center_y, sigma_y), min_y, max_y))
        rf_center_x = int(np.clip(np.random.normal(center_x, sigma_x), min_x, max_x))

        x1, x2 = rf_center_x - half, rf_center_x - half + receptive_field_size
        y1, y2 = rf_center_y - half, rf_center_y - half + receptive_field_size

        neuron_param_dict[i]["receptive_field"] = [[x1, y1], [x2, y2]]

        theta = np.deg2rad(orientation)

        sigma = 0.5 * wavelength
        # if orientation == 0:
        #     sigma *= 1.5
        # if orientation == 90:
        #     sigma *= 1.5
        neuron_param_dict[i]["kernel_even"] = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, wavelength, gamma, 0, ktype=cv2.CV_32F
        )
        neuron_param_dict[i]["kernel_odd"] = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, wavelength, gamma, np.pi / 2, ktype=cv2.CV_32F
        )

    print("Building spatial normalization matrix...")
    weight_matrix = _build_spatial_normalization_weights(neuron_param_dict, dn_params["spatial_sigma"])

    final_feature_matrix = np.zeros((len(images) * n_trials, n_neurons))
    start_time = time.time()

    print(f"Processing {len(images)} images across parallel workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(
                _process_single_image_divnorm,
                i, img, neuron_param_dict, weight_matrix, n_trials, fano_factor, sensor_noise_std, dn_params
            )
            for i, img in enumerate(images)
        ]

        for completed_count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            img_idx, img_features = future.result()

            start_row = img_idx * n_trials
            end_row = start_row + n_trials
            final_feature_matrix[start_row:end_row, :] = img_features

            elapsed_seconds = time.time() - start_time
            avg_time_per_img = elapsed_seconds / completed_count
            images_remaining = len(images) - completed_count
            eta_seconds = avg_time_per_img * images_remaining

            # Helper functions like format_seconds and scale_session assumed external
            print(
                f"Processed image {completed_count}/{len(images)} | "
                f"Time Taken: {format_seconds(elapsed_seconds)} | "
                f"ETA: {format_seconds(eta_seconds)}"
            )

    print("Scaling and saving features...")
    normalized_features = scale_session(final_feature_matrix)
    np.save(gabor_save_file, normalized_features)

    return pd.DataFrame(normalized_features)