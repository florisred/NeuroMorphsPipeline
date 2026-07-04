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

def format_seconds(secs):
    mins, secs = divmod(int(secs), 60)
    return f"{mins:02d}:{secs:02d}"

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