from data_objects.DataSource import DataSource
from MixIns.StimulusMixin import StimulusMixin
from pathlib import Path
import pandas as pd

class StimulusGaborDataSource(DataSource, StimulusMixin):
    def __init__(self, file_paths: list[Path], gabor_params: dict, output_dir: Path):
        super().__init__(file_paths)
        self.gabor_params = gabor_params
        self.output_dir = output_dir
        self.data_type = 'GaborStimulus'


    def load_data(self):
        images, images_names = self._load_images(
            image_dir=self.file_paths[0],
            flat=False
        )
        self.data = self._process_gabor(
            images=images,
            gabor_params=self.gabor_params,
            output_dir = self.output_dir
        )
        metadata = self._process_image_names(images_names)
        self.metadata.process_and_append(metadata)
        self.data.index = self.metadata.get_morph_names()
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        self.metadata.synchronize_with_data(self.data)



class StimulusPixelWiseDataSource(DataSource, StimulusMixin):
    def __init__(self, file_paths: list[Path]):
        super().__init__(file_paths)
        self.data_type = 'PixelWiseStimulus'


    def load_data(self):
        images_flat, images_names = self._load_images(
            image_dir=self.file_paths[0],
            flat=True
        )
        metadata = self._process_image_names(images_names)
        self.metadata.process_and_append(metadata)
        self.data = pd.DataFrame(images_flat)
        self.data.index = self.metadata.get_morph_names()
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        self.metadata.synchronize_with_data(self.data)

