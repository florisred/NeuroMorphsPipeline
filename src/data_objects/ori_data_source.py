from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals.array_api_extra import testing

from data_objects.data_source import DataSource
from data_objects.pca_data import PCAData
from data_objects.trial_metadata import TrialMetadata, OriMetaData
from utils.utils import load_h5_file, scale_session, load_images, ori_process_image_names, process_gabor, \
    create_distributed_gabor
import pandas as pd
import numpy.typing as npt

class OriDataSource(DataSource):
    def __init__(self, file_paths: list[Path]):
        super().__init__(file_paths)
        self._pca_data = None
        self._data_type = "OrientationData"
        self._pca_data = None
        self._labels_list = None
        #self.metadata=None

    def pca_data(self, n_components: int, return_data=True):
        if self._pca_data is None:
            pca_model = PCA(n_components=n_components)
            data = pca_model.fit_transform(self._data)
            data = pd.DataFrame(data, index=self._data.index)
            explained_variance = pca_model.explained_variance_ratio_
            self._pca_data = {'model': PCA,'data':data, 'explained_variance':explained_variance, 'n_components':n_components}
        elif self._pca_data['n_components'] != n_components:
            self._pca_data['model'] = PCA(n_components=n_components)
            self._pca_data['data'] =  pd.DataFrame(self._pca_data['model'].fit_transform(self._data), index=self._data.index)
            self._pca_data['n_components'] = n_components
            self._pca_data['explained_variance'] = self._pca_data['model'].explained_variance_ratio_


        self.pca_data_object=OriPCAData(
            pca_type=self._data_type,
            pca_output=self._pca_data['data'].to_numpy(),
            names_list=self.names_list,
            explained_variance=self._pca_data['explained_variance'],
        )

        if return_data: return self._pca_data['data']
        return None

    @property
    def explained_variance(self):
        if self.pca_data is None:
            raise AttributeError("No pca_data")
        return self._pca_data['explained_variance']

    @property
    def n_components(self):
        if self.pca_data is None:
            raise AttributeError("No pca_data")
        return self._pca_data['n_components']

    @property
    def names_list(self):
        return self._data.index.astype(str).to_list()




    def find_stimulus_cycles(self, n=3):
        return NotImplementedError("Finding stimulus cycles is not implemented in orientation tuning")

class OriTwoPhotonDataSource(OriDataSource):
    def __init__(self, file_paths: list[Path], data_location, metadata_locations):
        super().__init__(file_paths)
        self._data_type = "OrientationPixelWise"
        self._labels_list = metadata_locations
        self._data_location = data_location
        self._pca_data = None

    def load_data(self):
        data_dfs = []
        for session_dir in self.file_paths:
            raw_data_df, raw_meta_df = load_h5_file(session_dir, self._data_location, self._labels_list)
            meta_df = raw_meta_df.astype(float)
            data = raw_data_df.set_index(meta_df['orientation_deg'])
            data_sorted = data.sort_index()
            data_scaled = scale_session(data_sorted)
            data_dfs.append(data_scaled)
        pseudopopulation_df = pd.concat(data_dfs, axis=1)
        self._data = pseudopopulation_df


class OriPixeLWiseDataSource(OriDataSource):
    def __init__(self, file_paths: list[Path], output_dir: Path):
        super().__init__(file_paths)
        self._data_type = "OrientationPixelWise"

    def load_data(self):

        images_flat, images_names = load_images(
            image_dir=self.file_paths[0],
            flat=True
        )
        processed_names = ori_process_image_names(images_names)
        data = pd.DataFrame(images_flat, index=processed_names)
        self._data = data
        test=1

    def find_stimulus_cycles(self, n=3):
        return NotImplementedError("Finding stimulus cycles is not implemented in orientation tuning")

class OriGaborDataSource(OriDataSource):
    def __init__(self, file_paths: list[Path], gabor_params: dict, output_dir: Path):
        super().__init__(file_paths)
        self._data_type = "OrientationGabor"
        self.gabor_params = gabor_params
        self.output_dir = output_dir

    def load_data(self):
        images, images_names = load_images(
            image_dir=self.file_paths[0],
            flat=False
        )
        data = process_gabor(
            images=images,
            gabor_params=self.gabor_params,
            output_dir=self.output_dir
        )
        processed_names = ori_process_image_names(images_names)
        index = pd.Index(processed_names)
        data.set_index(index, inplace=True)
        self._data = data

class OriDistGaborDataSource(OriDataSource):
    def __init__(self, file_paths: list[Path], rf_dstr_path: Path, gabor_params: dict, output_dir: Path):
        super().__init__(file_paths)
        self._data_type = "OrientationGabor"
        self.gabor_params = gabor_params
        self.output_dir = output_dir
        self.rf_dstr_path = rf_dstr_path


    def load_data(self, n_neurons=500, n_trials = 10, rf_size_multiplier=1, save_and_load=True, rf_size_list: list[int] = None):
        gabor_params = self.gabor_params
        gabor_params['n_neurons'] = n_neurons

        images, images_names = load_images(
            image_dir=self.file_paths[0],
            flat=False,
        )
        processed_names = ori_process_image_names(images_names)
        images_names_duplicated = []
        for i_n in processed_names:
            for i in range(n_trials):
                images_names_duplicated.append(i_n)
        if rf_size_list is None:
            rf_dst_file = self.rf_dstr_path / 'rf_dstr.csv'
            rf_dstr = pd.read_csv(rf_dst_file)['RF_size_px']
            rf_dstr = rf_dstr * rf_size_multiplier
            print(rf_size_multiplier, np.mean(rf_dstr))
            rf_int = rf_dstr.astype(int)
            gabor_params['receptive_field_sizes'] = rf_int.to_list()
        else:
            gabor_params['receptive_field_sizes'] = rf_size_list

        data = create_distributed_gabor(images, gabor_params, self.output_dir, n_trials=n_trials, save_and_load=save_and_load)
        index = pd.Index(images_names_duplicated)
        data.set_index(index, inplace=True)
        self._data = data


class OriPCAData(PCAData):

    def __init__(self,pca_type: str, pca_output: npt.NDArray, names_list:list, explained_variance: npt.NDArray = None):
        metadata = OriMetaData(names_list)

        super().__init__(pca_type=pca_type, pca_output=pca_output, metadata=metadata, morph_names=pd.Index(names_list, name='morph_name'), explained_variance=explained_variance)



