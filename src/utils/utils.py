from itertools import combinations
from typing import Tuple

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
