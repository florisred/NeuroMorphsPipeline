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

        # Safe local scoping for orientation selection
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


def make_smooth_map(shape, coarse_size=8, smoothness=2.0, seed=None):
    """Generate a smooth 2D field over `shape` by upsampling coarse noise."""
    rng = np.random.default_rng(seed)
    coarse = rng.normal(size=(coarse_size, coarse_size))
    coarse = gaussian_filter(coarse, sigma=smoothness)
    zoom_y, zoom_x = shape[0] / coarse_size, shape[1] / coarse_size
    field = zoom(coarse, (zoom_y, zoom_x), order=3)
    field = field[: shape[0], : shape[1]]
    return field


def create_retinodivnorm_gabornet(
    images: np.ndarray, gabor_params: dict, output_dir: Path
) -> pd.DataFrame:
    gabor_save_file = output_dir / "RetinodivnormGaborNetCalculatedCache.npy"
    if not gabor_params["recalculate_gabornet"]:
        if gabor_save_file.exists():
            print("RetinodivnormGaborNet Feature Matrix found. Loading...")
            return pd.DataFrame(np.load(gabor_save_file))
    img_shape = images[0].shape
    gamma = gabor_params["gamma"]
    receptive_field_sizes = gabor_params["receptive_field_sizes"]
    n_neurons = gabor_params["n_neurons"]
    n_trials = gabor_params["n_trials"]
    fano_factor = gabor_params["fano_factor"]
    sensor_noise_std = gabor_params["sensor_noise_std"]

    # divisive normalization params (with sensible defaults if not provided)
    spatial_sigma = gabor_params.get("spatial_sigma", 110.0)
    orient_sigma = gabor_params.get("orient_sigma", 30.0)
    semisaturation = gabor_params.get("semisaturation", 0.1)
    norm_exponent = gabor_params.get("norm_exponent", 2.0)

    neuron_param_dict = {}
    # create a grid of neurons overlay over the images

    orientation_field_raw = make_smooth_map(
        img_shape, coarse_size=10, smoothness=1.5, seed=1
    )
    # map raw field (roughly N(0,1)) to angles 0-180 degrees
    orientation_field = orientation_field_raw - orientation_field_raw.min()
    orientation_field = orientation_field / orientation_field.max() * 180.0
    freq_field_raw = make_smooth_map(
        img_shape, coarse_size=10, smoothness=1.5, seed=2
    )
    grid_spacing = int(np.sqrt(img_shape[0] * img_shape[1] / n_neurons))
    grid_ys = np.arange(
        grid_spacing // 2, img_shape[0] - grid_spacing // 2, grid_spacing
    )
    grid_xs = np.arange(
        grid_spacing // 2, img_shape[1] - grid_spacing // 2, grid_spacing
    )
    grid_points = [(gy, gx) for gy in grid_ys for gx in grid_xs]
    if len(grid_points) > n_neurons:
        idx = np.random.choice(len(grid_points), n_neurons, replace=False)
        grid_points = [grid_points[k] for k in idx]
    elif len(grid_points) < n_neurons:
        extra = np.random.choice(
            len(grid_points), n_neurons - len(grid_points), replace=True
        )
        grid_points += [grid_points[k] for k in extra]

    jitter_std = grid_spacing * 0.25
    for i, (gy, gx) in enumerate(grid_points):
        neuron_param_dict[i] = {}

        rf_center_y = int(
            np.clip(gy + np.random.normal(0, jitter_std), 0, img_shape[0] - 1)
        )
        rf_center_x = int(
            np.clip(gx + np.random.normal(0, jitter_std), 0, img_shape[1] - 1)
        )

        # base preference comes from the smooth map at this location
        base_orientation = orientation_field[rf_center_y, rf_center_x]
        orientation = (
            base_orientation + np.random.normal(0, 10)
        ) % 180  # small per-neuron perturbation

        receptive_field_size = random.choice(receptive_field_sizes)
        half = receptive_field_size // 2
        rf_center_y = int(np.clip(rf_center_y, half, img_shape[0] - half))
        rf_center_x = int(np.clip(rf_center_x, half, img_shape[1] - half))

        x1, x2 = rf_center_x - half, rf_center_x - half + receptive_field_size
        y1, y2 = rf_center_y - half, rf_center_y - half + receptive_field_size
        neuron_param_dict[i]["receptive_field"] = [[x1, y1], [x2, y2]]
        neuron_param_dict[i]["orientation"] = orientation
        neuron_param_dict[i]["gamma"] = gamma
        rand = np.random.uniform(0, 1.0)
        if rand < 0.3:
            neuron_param_dict[i]["neuron_type"] = "simple"
        else:
            neuron_param_dict[i]["neuron_type"] = "complex"

        # base spatial frequency preference read from the smooth freq map,
        # used to bias wavelength selection so nearby neurons share SF tuning too
        freq_bias = freq_field_raw[rf_center_y, rf_center_x]
        cycles = np.clip(np.random.normal(2.9 + 0.4 * freq_bias, 0.3), 1.5, 5.0)
        wavelength = receptive_field_size / cycles
        neuron_param_dict[i]["wavelength"] = wavelength

        ksize = int(wavelength * 2) | 1
        if ksize > receptive_field_size:
            ksize = (
                receptive_field_size
                if (receptive_field_size % 2 == 1)
                else (receptive_field_size - 1)
            )
        ksize = max(ksize, 3)
        neuron_param_dict[i]["ksize"] = ksize

        theta = np.deg2rad(orientation)
        sigma_ratio = np.random.uniform(0.4, 0.65)
        sigma = sigma_ratio * wavelength
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

    print("Building divisive normalization pool...")
    normalization_pool_weights = build_normalization_pool(
        neuron_param_dict, spatial_sigma, orient_sigma
    )

    final_feature_matrix = np.zeros((len(images) * n_trials, n_neurons))
    start_time = time.time()

    print(f"Processing {len(images)} images across parallel workers...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _process_single_image_divnorm,
                i,
                img,
                neuron_param_dict,
                n_trials,
                fano_factor,
                sensor_noise_std,
                normalization_pool_weights,
                semisaturation,
                norm_exponent,
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


def build_normalization_pool(neuron_param_dict, spatial_sigma, orient_sigma):
    n = len(neuron_param_dict)
    centers = np.array([
        [
            (p["receptive_field"][0][1] + p["receptive_field"][1][1]) / 2,
            (p["receptive_field"][0][0] + p["receptive_field"][1][0]) / 2,
        ]
        for p in neuron_param_dict.values()
    ])
    orientations = np.array([p["orientation"] for p in neuron_param_dict.values()])

    spatial_dist = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=-1
    )
    spatial_w = np.exp(-(spatial_dist**2) / (2 * spatial_sigma**2))

    orient_diff = np.abs(orientations[:, None] - orientations[None, :])
    orient_diff = np.minimum(orient_diff, 180 - orient_diff)  # circular
    orient_w = np.exp(-(orient_diff**2) / (2 * orient_sigma**2))

    pool_weights = spatial_w * orient_w
    row_sums = pool_weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pool_weights = pool_weights / row_sums
    return pool_weights


def apply_divisive_normalization(
    raw_responses, pool_weights, semisaturation, exponent
):
    numerator = raw_responses**exponent
    pooled_drive = pool_weights @ (raw_responses**exponent)
    normalized = numerator / (pooled_drive + semisaturation**exponent)
    return normalized


def _process_single_image_divnorm(
    img_idx,
    img,
    neuron_param_dict,
    n_trials,
    fano_factor,
    sensor_noise_std,
    pool_weights,
    semisaturation,
    norm_exponent,
):
    n_neurons_local = len(neuron_param_dict)
    raw_responses = np.zeros(n_neurons_local)

    for j, neuron_params in neuron_param_dict.items():
        (x1, y1), (x2, y2) = neuron_params["receptive_field"]
        img_crop = img[y1:y2, x1:x2]

        if img_crop.size == 0:
            raise AssertionError("Cropped image does not contain pixels")

        res_even = cv2.filter2D(img_crop, cv2.CV_32F, neuron_params["kernel_even"])
        res_odd = cv2.filter2D(img_crop, cv2.CV_32F, neuron_params["kernel_odd"])
        neuron_type = neuron_params["neuron_type"]
        if neuron_type == "simple":
            magnitude = np.abs(res_even)
        else:
            magnitude = np.sqrt(res_even**2 + res_odd**2)
        raw_responses[j] = np.mean(magnitude)

    normalized_responses = apply_divisive_normalization(
        raw_responses, pool_weights, semisaturation, norm_exponent
    )

    img_features = np.zeros((n_trials, n_neurons_local))
    for j in range(n_neurons_local):
        mu = normalized_responses[j] * 10
        if mu > 0:
            poisson_counts = np.random.poisson(mu / fano_factor, size=n_trials)
            trial_activations = poisson_counts.astype(float) * fano_factor
        else:
            trial_activations = np.zeros(n_trials, dtype=float)


        noise = np.random.normal(0, sensor_noise_std, size=n_trials)
        img_features[:, j] = np.maximum(0, trial_activations + noise)

    return img_idx, img_features