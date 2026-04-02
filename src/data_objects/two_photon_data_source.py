from pathlib import Path

from scipy.interpolate import interp1d

from data_objects.trial_metadata import TrialMetadata
from data_objects.data_source import DataSource
from MixIns.two_photon_mixin import TwoPhotonMixIn
import pandas as pd
from utils.utils import scale_session, split_morphs
import logging
import numpy as np


logger = logging.getLogger(__name__)


class TwoPhotonDataSource(DataSource, TwoPhotonMixIn):

    def __init__(self, file_paths: list[Path], data_location: str):
        super().__init__(file_paths)
        self._data_location = data_location
        self._labels_list = ['pair_key', 'step_index', 'src_cat', 'dst_cat']
        self._data_type = 'TwoPhoton'

    def load_data(self, split: bool = False, seed: int = 42):
        """
        Loads the .h5 data from each session and loads them in the metadata file.
        Then, it isolates shared morphs across sessions and interpolates repetitions
        so that every session matches the maximum repetition count for a given morph.
        """
        data_dfs = []
        self._split = split
        for session_dir in self.file_paths:
            raw_data_df, raw_meta_df = self._load_h5_file(session_dir, self._data_location, self._labels_list)
            temp_meta = TrialMetadata()
            temp_meta.process_and_append(raw_meta_df)
            raw_morph_names = temp_meta.morph_names
            temp_meta_df = temp_meta._metadata_df
            raw_data_df.index = raw_morph_names
            if split: raw_data_df, temp_meta_df = split_morphs(raw_trials=raw_data_df, raw_metadata=temp_meta_df,
                                                               seed=seed)
            data_dfs.append(raw_data_df)
            self._metadata.append(temp_meta_df)

        shared_morphs = self.metadata.shared_morphs

        # Step 1: Filter and scale sessions first
        processed_sessions = []
        for data_df in data_dfs:
            filtered_df = data_df[data_df.index.isin(shared_morphs)]
            scaled_session = scale_session(filtered_df)
            processed_sessions.append(scaled_session)

        # Step 2: Find the maximum repetitions for each morph across all sessions
        max_reps_per_morph = {}
        for p_df in processed_sessions:
            rep_counts = p_df.index.value_counts()
            for morph, count in rep_counts.items():
                max_reps_per_morph[morph] = max(max_reps_per_morph.get(morph, 0), count)

        # Helper function to linearly interpolate trials
        def interpolate_trials(morph_df, target_reps):
            n_current = len(morph_df)
            if n_current == target_reps:
                return morph_df.reset_index(drop=True)
            if n_current == 1:
                # Can't interpolate a single trial, so we just duplicate it
                return pd.concat([morph_df] * target_reps).reset_index(drop=True)

            # Create a relative timeline from 0 to 1 for both current and target lengths
            old_idx = np.linspace(0, 1, n_current)
            new_idx = np.linspace(0, 1, target_reps)

            temp_df = morph_df.copy()
            temp_df.index = old_idx

            # Reindex to combine both timelines, then interpolate the missing values
            combined_idx = np.union1d(old_idx, new_idx)
            interp_df = temp_df.reindex(combined_idx).interpolate(method='index')

            # Extract only the target timeline
            result_df = interp_df.loc[new_idx].copy()
            result_df.index = range(target_reps)
            return result_df

        # Step 3: Interpolate each morph per session
        aligned_data_dfs = []
        for p_df in processed_sessions:
            session_morph_dfs = []
            # We sort shared_morphs to ensure exact row alignment between sessions
            for morph in sorted(list(shared_morphs)):
                # Extract all repetitions for this specific morph
                morph_df = p_df.loc[[morph]]
                target_reps = max_reps_per_morph[morph]

                # Interpolate to match max repetitions
                interp_df = interpolate_trials(morph_df, target_reps)
                interp_df['morph_name'] = morph
                session_morph_dfs.append(interp_df)

            # Combine all interpolated morphs back into a single session dataframe
            session_aligned = pd.concat(session_morph_dfs, ignore_index=True)
            aligned_data_dfs.append(session_aligned)

        # Step 4: Drop the 'morph_name' string column from all but the first session
        for i in range(len(aligned_data_dfs)):
            if i == 0: continue
            aligned_data_dfs[i] = aligned_data_dfs[i].drop(columns='morph_name')

        # Step 5: Concatenate side-by-side.
        # (Since we standardized lengths and morph order, rows match perfectly)
        combined_df = pd.concat(aligned_data_dfs, axis=1)
        combined_df.set_index(keys='morph_name', inplace=True)

        self._metadata.synchronize_with_data(combined_df)
        self._data = combined_df
        logger.info(f"Loaded {len(combined_df)} morph trials with {combined_df.shape[1]} neurons")

    def load_data_old(self, split:bool=False, seed:int=42):
        """
        Loads the .h5 data from each session and loads them in the metadata file.
        Then, it groups the data from trials with identical stimuli together and averages the mean activation
        """
        data_dfs = []
        self._split = split
        for session_dir in self.file_paths:
            raw_data_df, raw_meta_df = self._load_h5_file(session_dir, self._data_location, self._labels_list)
            temp_meta = TrialMetadata()
            temp_meta.process_and_append(raw_meta_df)
            raw_morph_names = temp_meta.morph_names
            temp_meta_df = temp_meta._metadata_df
            raw_data_df.index = raw_morph_names
            if split: raw_data_df, temp_meta_df = split_morphs(raw_trials=raw_data_df, raw_metadata=temp_meta_df, seed=seed)
            data_dfs.append(raw_data_df)
            self._metadata.append(temp_meta_df)

        raw_data_dfs = []
        shared_morphs = self.metadata.shared_morphs
        for data_df in data_dfs:
            filtered_df = data_df[data_df.index.isin(shared_morphs)]
            scaled_session = scale_session(filtered_df)
            sorted_scaled_session = scaled_session.sort_index()
            sorted_scaled_session.reset_index(inplace=True, drop=False)
            raw_data_dfs.append(sorted_scaled_session)
        for i in range(len(raw_data_dfs)):
            if i == 0: continue
            raw_data_dfs[i] = raw_data_dfs[i].drop(columns='morph_name')
        combined_df = pd.concat(raw_data_dfs, axis=1, join='inner')
        combined_df.set_index(keys='morph_name', inplace=True)
        self._metadata.synchronize_with_data(combined_df)
        self._data = combined_df
        logger.info(f"Loaded {len(combined_df)} morphs with {combined_df.shape[1]} neurons")

