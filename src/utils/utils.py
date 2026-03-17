from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def scale_session(X):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

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
