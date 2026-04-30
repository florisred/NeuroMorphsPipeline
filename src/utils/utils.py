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


def split_morphs(
        raw_trials: pd.DataFrame,
        raw_metadata: pd.DataFrame,
        train_percent: float = 0.7,
        seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Splits the trials into train and test sets, and sets their morph name accordingly
    :param raw_trials: raw trials loaded directly from h5 file
    :param raw_metadata: metadata after being processed by a TrialMetaData() object, in dataframe format
    :param train_percent: the percentage of training set
    :param seed: the seed to use randomizer
    :return: Tuple[split_trials, split_metadata]
    """

    if not 0 < train_percent <= 1: raise ValueError("test_procent must be between 0 and 1")
    shuffled_trials, shuffled_metadata = shuffle(raw_trials, raw_metadata, seed)

    counts = Counter()
    total_counts = Counter(shuffled_metadata.index)
    labels = []
    for morph in shuffled_metadata.index:
        counts[morph] += 1
        limit = round(total_counts[morph] * train_percent)
        tag = "train" if counts[morph] <= limit else "test"
        labels.append(f"{morph}_{tag}")

    shuffled_metadata.index = labels
    shuffled_metadata['morph_name'] = labels
    shuffled_trials.index = labels
    shuffled_metadata.rename_axis('morph', inplace=True)
    shuffled_trials.rename_axis('morph_name', inplace=True)
    return shuffled_trials, shuffled_metadata




def shuffle(
        raw_data_df: pd.DataFrame,
        raw_metadata_df: pd.DataFrame,
        seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Shuffles the raw data and metadata with a set seed
    :param raw_data_df: raw trials loaded directly from h5 file
    :param raw_metadata_df: metadata after being processed by a TrialMetaData() object, in dataframe format
    :param seed:
    :return:
    """
    if len(raw_metadata_df) != len(raw_data_df): raise ValueError("Data and Metadata row counts do not match")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(raw_metadata_df))
    return raw_data_df.iloc[permutation], raw_metadata_df.iloc[permutation]


def find_max_separation(pca_data_dict, num_comps) -> list:
    dist_dict = {}
    for name, pca_data in pca_data_dict.items():
        dist_dict[name] = {}
        anchor_data = pca_data.anchors.drop_duplicates()
        for component_combination in list(combinations(range(anchor_data.shape[1]), num_comps)):
            anchor_data_filtered_components = anchor_data[list(component_combination)]
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

 def create_distributed_gabor(images: npt.NDArray, gabor_params: dict, output_dir: Path, n_trials = 7) -> pd.DataFrame:
        gabor_save_file = output_dir / "gabor_distributed_normalized_features_stimuli.npy"
        if Path.exists(gabor_save_file):
            print("Gabor Feature matrix already exists. Loading...")
            return pd.DataFrame(np.load(gabor_save_file))
        wavelengths = gabor_params["wavelengths"]
        orientations = gabor_params["orientations"]
        gamma = gabor_params["gamma"]
        receptive_field_sizes = gabor_params["receptive_field_sizes"]
        n_neurons = gabor_params["n_neurons"]
        fano_factor = 8
        sensor_noise_std = 0.50
        neuron_param_dict = {}
        for i in range(n_neurons):
            neuron_param_dict[i] = {}
            neuron_param_dict[i]["orientation"] = random.choice(orientations)
            neuron_param_dict[i]["wavelength"] = random.choice(wavelengths)
            neuron_param_dict[i]["gamma"] = gamma
            while True:
                receptive_field_size = random.choice(receptive_field_sizes) * 2
                img_shape = images[0].shape
                if receptive_field_size < min(img_shape) / 2: break
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
            if i % 10 == 0:
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
        np.save(gabor_save_file, normalized_features)

        return pd.DataFrame(normalized_features)


def process_image_names(image_names_raw):
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

def process_gabor(images: npt.NDArray, gabor_params: dict, output_dir: Path) -> pd.DataFrame:
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


def load_images(image_dir: Path, flat: bool = False) -> tuple[npt.NDArray[np.float32], list[str]]:
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

def load_metadata_2p(f, labels_list: list[str]):
    raw_metadata_dataframe = pd.DataFrame()
    for metadata_location in labels_list:
        meta_name = metadata_location.split("/")[-1]
        metadata_array = np.array(f['y'][metadata_location]).flatten()
        raw_metadata_dataframe[meta_name] = metadata_array.astype(str)
    return raw_metadata_dataframe

def load_h5_file(session_dir: Path, data_location: str, labels_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads data from h5 file
    :param session_dir:
    :param data_location:
    :param labels_list:
    :return: Tuple[raw_data_df: pd.DataFrame, raw_metadata_df: pd.DataFrame]
    """
    two_photon_folder = session_dir / '2p_data'
    file = list(two_photon_folder.glob("*.h5"))
    file = [f for f in file if not f.name.startswith('.')]
    if len(file) != 1: raise AssertionError(f"Need exactly one .hy file in {two_photon_folder}")
    f = h5py.File(file[0], 'r')
    raw_data_df = pd.DataFrame(f[data_location]).dropna(axis=0, how='all')
    raw_metadata_df = load_metadata_2p(f, labels_list)
    raw_data_df = raw_data_df.T
    return raw_data_df, raw_metadata_df