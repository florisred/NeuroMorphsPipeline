from pathlib import Path
from data_loader.TrialMetadata import TrialMetadata
from data_loader.DataSource import DataSource
from data_loader.MixIns.TwoPhotonMixIn import TwoPhotonMixIn
from src.utils import scale_session
import pandas as pd


class TwoPhotonDataSource(DataSource, TwoPhotonMixIn):

    def __init__(self, file_paths: list[Path], data_location: str):
        super().__init__(file_paths)
        self.data_location = data_location
        self.labels_list = ['pair_key', 'step_index', 'src_cat', 'dst_cat']

    def load_data(self):
        """
        Loads the .h5 data from each session and loads them in the metadata file.
        Then, it groups the data from trials with identical stimuli together and averages the mean activation
        """
        data_dfs = []
        for session_dir in self.file_paths:
            raw_data_df, raw_meta_df = self._load_h5_file(session_dir, self.data_location, self.labels_list)
            temp_meta = TrialMetadata()
            temp_meta.process_and_append(raw_meta_df)
            grouped_data = raw_data_df.groupby(temp_meta.get_morph_names()).mean()
            data_dfs.append(grouped_data)
            self.metadata.process_and_append(raw_meta_df)

        #concatinate the data
        combined_df = pd.concat(data_dfs, axis=1, join='inner')
        self.metadata.synchronize_with_data(combined_df)
        self.data = combined_df




