import ctypes
import random
from itertools import combinations
from pathlib import Path
from typing import Tuple

import cv2
import h5py
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import numpy.typing as npt
from collections import Counter, defaultdict


def scale_session(X) -> pd.DataFrame | npt.ArrayLike:
    """
    :param X: data to be scaled, either a pandas DataFrame or a numpy array
    :return: the scaled da ta in the original format
    """
    scaler = StandardScaler()
    if type(X) == pd.DataFrame:
        return pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    elif type(X) == np.ndarray:
        return scaler.fit_transform(X)
    else:
        raise TypeError("Please provide a pandas DataFrame or numpy array")

def create_name_from_list(
        transition_list: list,
        first_part: str ='subset'
) -> str:
    """
    Simple helper function to create a name (for the plot) from a transition list with A__B
    :param transition_list:
    :param first_part:
    :return: The name to be used for the subset
    """
    all_anchs = []
    for trans in transition_list:
        all_anchs.append(trans.split('__'))
    unique_anchs = np.unique(all_anchs)
    final_name = first_part
    for anchor in unique_anchs:
        final_name += f'-{anchor}'
    return final_name


def find_max_separation(num_comps, pca_data_dict, filter=None):
    dist_dict = {}
    for name, pca_data in pca_data_dict.items():
        if filter is not None and filter not in name:
            continue
        dist_dict[name] = {}
        data_to_use = pca_data.anchors.drop_duplicates()
        for component_combination in list(combinations(range(data_to_use.shape[1]), num_comps)):
            anchor_data_filtered_components = data_to_use[list(component_combination)]
            all_distances = pdist(anchor_data_filtered_components.values)
            total_separation = all_distances.mean()
            dist_dict[name][component_combination] = total_separation
    comp_dict = {}
    for name, distances in dist_dict.items():
        max_key = None
        for key in distances.keys():
            if max_key is None or distances[key] > distances[max_key]:
                max_key = key
        comp_dict[name] = max_key
    return comp_dict

def find_max_seperation_dataframe(num_comps, pca_dataframe: pd.DataFrame):
    index=pca_dataframe.index
    data_to_use = pca_dataframe.groupby(index).mean()
    dist_dict = {}
    for component_combination in list(combinations(range(data_to_use.shape[1]), num_comps)):
        data_filtered_components = data_to_use[list(component_combination)]
        all_distances = pdist(data_filtered_components.values)
        total_separation = all_distances.mean()
        dist_dict[component_combination] = total_separation
    max_key = None
    for key in dist_dict.keys():
        if max_key is None or dist_dict[key] > dist_dict[max_key]:
            max_key = key
    return max_key



def create_distributed_gabor(images: npt.NDArray, gabor_params: dict, output_dir: Path, n_trials = 7, save_and_load=True) -> pd.DataFrame:
        if save_and_load:
            gabor_save_file = output_dir / "gabor_distributed_normalized_features_stimuli.npy"
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
        print("Scaling and saving features...")
        normalized_features = scale_session(final_feature_matrix)
        if save_and_load:
            np.save(gabor_save_file, normalized_features)

        return pd.DataFrame(normalized_features)


def process_image_names(image_names_raw):
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


def load_images(image_dir: Path, flat: bool = False) -> tuple[npt.NDArray[np.float32], list[str]]:
    """
    Loads PNG images from a directory, normalizes them to [0, 1],
    and optionally flattens them.
    """
    images = []
    image_names = []
    image_paths_png = sorted(image_dir.glob("*.png"))
    image_paths_bmp = sorted(image_dir.glob("*.bmp"))
    image_paths = image_paths_png + image_paths_bmp
    for image_path in image_paths:
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

def load_metadata_2p(f, labels_list: list[str]):
    """
    helper function of load_h5_file. the loaded file, and the labels, and returns a dataframe with the correct columns
    and data
    """
    raw_metadata_dataframe = pd.DataFrame()
    for metadata_location in labels_list:
        f_metadata, meta_name = browse_h5_file(metadata_location, f)
        metadata_array = np.array(f_metadata).flatten()
        raw_metadata_dataframe[meta_name] = metadata_array.astype(str)
    return raw_metadata_dataframe

def load_h5_file(session_dir: Path, data_location: str, labels_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads data from h5 file located in session_dir. Looks within that file for the data in data_location, and for
    metadata in each entry of lables_list.


    :param session_dir:
    :param data_location:
    :param labels_list:
    :return: Tuple[raw_data_df: pd.DataFrame, raw_metadata_df: pd.DataFrame]
    """
    h5_folder = session_dir
    file = list(h5_folder.glob("*.h5"))
    file = [f for f in file if not f.name.startswith('.')]
    if len(file) != 1: raise AssertionError(f"Need exactly one .hy file in {h5_folder}")
    f = h5py.File(file[0], 'r')
    x_h5, loc = browse_h5_file(data_location, f)
    raw_data_df = pd.DataFrame(x_h5).dropna(axis=0, how='all')
    raw_metadata_df = load_metadata_2p(f, labels_list)
    raw_data_df = raw_data_df.T
    return raw_data_df, raw_metadata_df

def browse_h5_file(data_location: str, file):
    try:
        data_split = data_location.split("/")
        fx=file
        for loc in data_split:
            fx=fx[loc]
        return fx, loc
    except Exception as e:
        raise ValueError(
            f"Could not find the label {data_location} in the h5 file.\n Make sure to split the datasets with a '/'. \nf{e}")

def ori_process_image_names(image_names:list):
    """
    :param image_names: List of image paths. Has to have '_' as a divider, one part has to be parsable as an int between 0 and 180, representing degrees of orientation
    """
    if len(image_names) ==0: raise AssertionError("Need at least one image to parse")
    int_names = []
    for name in image_names:
        img_parts = name.split('_')
        for part in img_parts:
            if '.bmp' in part:
                part = part[:-4]
            try:
                int_part = float(part)
                int_names.append(int_part)
            except ValueError: test=1
    return int_names


def mark_uniques(input_list) -> list:
    """
    Function that takes a list, and marks each first occurrence of each value as True, the rest as false. Returns a list
    of the same shape as the original list, with only True or False

    :param input_list: list that should be marked for first occurences
    """
    seen = set()
    result = []

    for item in input_list:
        if item not in seen:
            result.append(True)
            seen.add(item)
        else:
            result.append(False)

    return result