from data_objects.data_source import DataSource
from MixIns.stimulus_mixin import StimulusMixin
from pathlib import Path
import pandas as pd

class StimulusGaborDataSource(DataSource, StimulusMixin):
    def __init__(self, file_paths: list[Path], gabor_params: dict, output_dir: Path):
        super().__init__(file_paths)
        self.gabor_params = gabor_params
        self.output_dir = output_dir
        self._data_type = 'GaborStimulus'


    def load_data(self):
        images, images_names = self._load_images(
            image_dir=self.file_paths[0],
            flat=False
        )
        self._data = self._process_gabor(
            images=images,
            gabor_params=self.gabor_params,
            output_dir = self.output_dir
        )
        metadata = self._process_image_names(images_names)
        self._metadata.process_and_append(metadata)
        self._data.index = self._metadata.morph_names
        self._data = self._data[~self.data.index.duplicated(keep='first')]
        self._metadata.synchronize_with_data(self._data)



class StimulusPixelWiseDataSource(DataSource, StimulusMixin):
    def __init__(self, file_paths: list[Path]):
        super().__init__(file_paths)
        self._data_type = 'PixelWiseStimulus'


    def load_data(self):
        images_flat, images_names = self._load_images(
            image_dir=self.file_paths[0],
            flat=True
        )
        metadata = self._process_image_names(images_names)
        self._metadata.process_and_append(metadata)
        self._data = pd.DataFrame(images_flat)
        self._data.index = self._metadata.morph_names
        self._data = self.data[~self._data.index.duplicated(keep='first')]
        self._metadata.synchronize_with_data(self.data)

class DistributedGaborDataSource(DataSource, StimulusMixin):
    def __init__(self, file_paths: list[Path], gabor_params: dict, output_dir: Path):
        super().__init__(file_paths)
        self.gabor_params = gabor_params
        self.output_dir = output_dir
        self._data_type = 'DistributedGaborStimulus'

    def load_data(self, n_neurons=500, n_trials = 7):
        gabor_params = self.gabor_params
        gabor_params['n_neurons'] = n_neurons
        images, images_names = self._load_images(
            image_dir=self.file_paths[0],
            flat=False
        )
        images_names_duplicated = []
        for i_n, img in zip(images_names, images):
            for i in range(n_trials):
                images_names_duplicated.append(i_n)
        self._data = self._process_distributed_gabor(images, gabor_params, self.output_dir, n_trials=7)
        metadata = self._process_image_names(images_names_duplicated)

        self._metadata.process_and_append(metadata)

        self._data.index = self._metadata.morph_names
        self._metadata.synchronize_with_data(self._data)


