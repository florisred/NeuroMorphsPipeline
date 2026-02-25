from os.path import join
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Twophoton:
    def __init__(self):
        self.metadata_df = None
        self.data_df = None
        self.labels = None
        self.processed_df = None
        self.pca_coords=None
        self.variance_explained=None
        self.metadata_list = [
                "meta/mouse",
                "meta/date",
                "meta/sess_repN",
                "meta/sess_index"
            ]
        self.labels_list = ["y/pair_key", "y/step_index"]





    def load_2p_data(self, data_dir, data_location, label_location):
        two_photon_folder = join(data_dir, "2p_data")
        file = list(Path(two_photon_folder).glob("*.h5"))
        if len(file) != 1: raise AssertionError("Need exactly one .hy file")
        file = file[0]
        f = h5py.File(file, 'r')
        self.data_df = pd.DataFrame(f[data_location])
        self.labels = self._load_list(f, self.labels_list)
        #metadata_df = self._load_list(f, self.metadata_list)


        #self.metdata = metadata_df



    def calc_mean_per_stimulus_and_scale(self):
        processed_df = self.data_df.T.groupby(self.labels.iloc[:, 0]).mean()
        #processed_df.reset_index(inplace=True)
        scaler = StandardScaler()
        processed_df = scaler.fit_transform(processed_df)
        self.processed_df = processed_df



    def peform_pca(self):
        pca = PCA(n_components=8)
        self.pca_coords = pca.fit_transform(self.processed_df)
        self.variance_explained = pca.explained_variance_ratio_
        return self.pca_coords, self.variance_explained, np.unique(self.labels)


    @staticmethod
    def _load_list(f, metadata_list):
        metadata_dataframe = pd.DataFrame()
        for metadata_location in metadata_list:
            meta_name = metadata_location.split("/")[-1]
            metadata_array = np.array(f[metadata_location]).flatten()
            metadata_dataframe[meta_name] = metadata_array.astype(str)
        metadata_dataframe = metadata_dataframe.T
        metadata_df = metadata_dataframe.T
        col1 = metadata_df.iloc[:, 0].astype(str)
        col2 = metadata_df.iloc[:, 1].astype(str).str.zfill(2)
        metadata_df['combined'] = col1 + col2
        metadata_df = metadata_df[['combined']]
        return metadata_df