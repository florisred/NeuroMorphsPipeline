from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from collections import Counter


def scale_session(X):
    scaler = StandardScaler()
    if type(X) == pd.DataFrame:
        return pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    elif type(X) == np.ndarray:
        return scaler.fit_transform(X)
    else:
        raise TypeError("Please provide a pandas DataFrame or numpy array")

def calc_mean_per_stimulus(data_df, labels):
    grouped_data = data_df.T.groupby(labels["morph_name"].values).mean()
    grouped_data_labels = labels.drop_duplicates(subset='morph_name').sort_values(by=['morph_name'])
    grouped_data.index = grouped_data_labels['morph_name']
    return grouped_data, grouped_data_labels

def create_name_from_list(transition_list: list, first_part: str ='subset'):
    all_anchs = []
    for trans in transition_list:
        all_anchs.append(trans.split('__'))
    unique_anchs = np.unique(all_anchs)
    final_name = first_part
    for anchor in unique_anchs:
        final_name += f'-{anchor}'
    return final_name


def split_morphs(raw_trials: pd.DataFrame, raw_metadata: pd.DataFrame, train_percent: float = 0.7, seed: int = 42):
    if not 0 < train_percent <= 1: raise ValueError("test_procent must be between 0 and 1")
    shuffled_trials, shuffled_metadata, permutation = shuffle(raw_trials, raw_metadata, seed)

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




def shuffle(raw_data_df: pd.DataFrame, raw_metadata_df: pd.DataFrame, seed: int = 42):
    if len(raw_metadata_df) != len(raw_data_df): raise ValueError("Data and Metadata row counts do not match")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(raw_metadata_df))
    return raw_data_df.iloc[permutation], raw_metadata_df.iloc[permutation], permutation

