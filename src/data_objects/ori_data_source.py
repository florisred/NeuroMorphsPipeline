from pathlib import Path
from data_objects.data_source import DataSource
from data_objects.trial_metadata import TrialMetadata, OriMetaData
from utils.utils import load_h5_file, scale_session
import pandas as pd


class OriTwoPhotonDataSource(DataSource):
    def __init__(self, file_paths: list[Path]):
        super().__init__(file_paths)
        self._data_type = "OrientationPixelWise"
        self._labels_list = ['orientation_deg']
        self._data_location = 'X'

    def load_data(self):
        data_dfs = []
        for session_dir in self.file_paths:
            raw_data_df, raw_meta_df = load_h5_file(session_dir, self._data_location, self._labels_list)
            meta = TrialMetadata(raw_data_df)
            meta_df = raw_meta_df.astype(float)
            data = raw_data_df.set_index(meta_df['orientation_deg'])
            data_sorted = data.sort_index()
            data_scaled = scale_session(data_sorted)
            data_dfs.append(data_scaled)
        pseudopopulation_df = pd.concat(data_dfs, axis=1)
        self._data = pseudopopulation_df
        pass

    def find_stimulus_cycles(self, n=3):
        return NotImplementedError("Finding stimulus cycles is not implemented in orientation tuning")

class OriPixeLWiseDataSource(DataSource):
    def __init__(self, file_paths: list[Path], output_dir: Path):
        super().__init__(file_paths)
        self._data_type = "OrientationPixelWise"

    def load_data(self):
        pass

    def find_stimulus_cycles(self, n=3):
        return NotImplementedError("Finding stimulus cycles is not implemented in orientation tuning")