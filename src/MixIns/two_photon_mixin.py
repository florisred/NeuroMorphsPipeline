from pathlib import Path
import numpy as np
import pandas as pd

import h5py

class TwoPhotonMixIn:

    def _load_h5_file(self, session_dir: Path, data_location: str, labels_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads data from h5 file
        :param session_dir:
        :param data_location:
        :param labels_list:
        :return: Tuple[raw_data_df: pd.DataFrame, raw_metadata_df: pd.DataFrame]
        """
        two_photon_folder = session_dir / '2p_data'
        file = list(two_photon_folder.glob("*.h5"))
        file = [f for f in file if not f.name.startswith('.')]
        if len(file) != 1: raise AssertionError(f"Need exactly one .hy file in {two_photon_folder}")
        f = h5py.File(file[0], 'r')
        raw_data_df = pd.DataFrame(f[data_location]).dropna(axis=0, how='all')
        raw_metadata_df = self._load_metadata_2p(f, labels_list)

        # since in the raw the rows indicate neurons and the columns the trials, we want to transpose this
        raw_data_df = raw_data_df.T
        return raw_data_df, raw_metadata_df

    @staticmethod
    def _load_metadata_2p(f, labels_list: list[str]):
        raw_metadata_dataframe = pd.DataFrame()
        for metadata_location in labels_list:
            meta_name = metadata_location.split("/")[-1]
            metadata_array = np.array(f['y'][metadata_location]).flatten()
            raw_metadata_dataframe[meta_name] = metadata_array.astype(str)

        return raw_metadata_dataframe

