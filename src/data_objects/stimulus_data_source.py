from data_objects.data_source import DataSource
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.utils import process_image_names, load_images
from src.helper.gabor_helper import create_distributed_gabor, process_gabor

class StimulusGaborDataSource(DataSource):
    def __init__(self, file_paths: list[Path], gabor_params: dict, output_dir: Path):
        super().__init__(file_paths)
        self.gabor_params = gabor_params
        self.output_dir = output_dir
        self._data_type = 'GaborStimulus'


    def load_data(self):
        images, images_names = load_images(
            image_dir=self.file_paths[0],
            flat=False
        )
        self._data = process_gabor(
            images=images,
            gabor_params=self.gabor_params,
            output_dir = self.output_dir
        )
        metadata = process_image_names(images_names)
        self._metadata.process_and_append(metadata)
        self._data.index = self._metadata.morph_names
        self._data = self._data[~self.data.index.duplicated(keep='first')]
        self._metadata.synchronize_with_data(self._data)



class StimulusPixelWiseDataSource(DataSource):
    def __init__(self, file_paths: list[Path]):
        super().__init__(file_paths)
        self._data_type = 'PixelWiseStimulus'


    def load_data(self):
        images_flat, images_names = load_images(
            image_dir=self.file_paths[0],
            flat=True
        )
        metadata = process_image_names(images_names)
        self._metadata.process_and_append(metadata)
        self._data = pd.DataFrame(images_flat)
        self._data.index = self._metadata.morph_names
        self._data = self._data[~self._data.index.duplicated(keep='first')]
        self._metadata.synchronize_with_data(self.data)

class DistributedGaborDataSource(DataSource):
    def __init__(self, file_paths: list[Path], rf_dstr_path: Path, gabor_params: dict, output_dir: Path):
        super().__init__(file_paths)
        self.rf_dstr_path = rf_dstr_path
        self.gabor_params = gabor_params
        self.output_dir = output_dir
        self._data_type = 'DistributedGaborStimulus'

    def load_data(self, n_neurons=5000, n_trials = 7, rf_size_multiplier=1, save_and_load=True, rf_size_list: list[int] = None):

        gabor_params = self.gabor_params
        gabor_params['n_neurons'] = n_neurons
        if rf_size_list is None:
            rf_dst_file = self.rf_dstr_path / 'rf_dstr.csv'
            rf_dstr = pd.read_csv(rf_dst_file)['RF_size_px']
            rf_dstr = rf_dstr * rf_size_multiplier
            print(rf_size_multiplier, np.mean(rf_dstr))
            rf_int = rf_dstr.astype(int)
            gabor_params['receptive_field_sizes'] = rf_int.to_list()
        else:
            gabor_params['receptive_field_sizes'] = rf_size_list

        images, images_names = load_images(
            image_dir=self.file_paths[0],
            flat=False
        )
        images_names_duplicated = []
        for i_n in images_names:
            for i in range(n_trials):
                images_names_duplicated.append(i_n)
        self._data = create_distributed_gabor(images, gabor_params, self.output_dir, n_trials=n_trials, save_and_load=save_and_load)
        metadata = process_image_names(images_names_duplicated)

        self._metadata.process_and_append(metadata)

        self._data.index = self._metadata.morph_names
        self._metadata.synchronize_with_data(self._data)

