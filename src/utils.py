import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_h5_file(file_location, data_location, label_location, metadata_list):
    f=h5py.File(file_location, 'r')
    metadata_df = load_metadata(f, metadata_list)

    return pd.DataFrame(f[data_location]), np.array(f[label_location]).flatten(), metadata_df

def load_metadata(f, metadata_list):
    metadata_dataframe = pd.DataFrame()
    for metadata_location in metadata_list:
        meta_name = metadata_location[5:]
        metadata_array = np.array(f[metadata_location]).flatten()
        metadata_dataframe[meta_name] = metadata_array.astype(str)
    return metadata_dataframe

def scale_session(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)