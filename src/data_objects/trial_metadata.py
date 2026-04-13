import pandas as pd
import numpy as np
import numpy.typing as npt
import copy
from typing import Union, List, Tuple, Optional, Set


class TrialMetadata:
    """
    Manages and filters metadata for experimental trials, supporting masking,
    session synchronization, and morph name generation.
    """

    def __init__(self, metadata_df: Optional[pd.DataFrame] = None):
        """
        Initializes the TrialMetadata object.

        Args:
            metadata_df: Initial metadata. Defaults to an empty DataFrame.
        """
        self._metadata_df: pd.DataFrame = metadata_df if metadata_df is not None else pd.DataFrame()
        self._trial_lens: List[int] = []
        self._masked_metadata: Optional[pd.DataFrame] = None
        self._use_mask: bool = False
        self._train_mask: Optional[np.ndarray] = None
        self._filter_mask: Optional[np.ndarray] = None

    @property
    def shared_morphs(self) -> List[str]:
        """Finds morph names that appear in every recorded session."""
        morphs_per_session: List[Set[str]] = []
        start_idx = 0

        # Use full metadata for this to ensure session integrity
        full_morphs = self._metadata_df['morph_name']

        for length in self._trial_lens:
            session_morphs = set(full_morphs.iloc[start_idx: start_idx + length].unique())
            morphs_per_session.append(session_morphs)
            start_idx += length

        shared = set.intersection(*morphs_per_session) if morphs_per_session else set()
        return list(shared)


    def process_and_append(self, raw_trials_metadata_df: pd.DataFrame) -> None:
        """
        Processes raw session metadata and appends it to the global storage.

        Calculates normalized steps, identifies anchors vs morphs, and
        constructs unique 'morph_names' based on source and destination categories.
        """
        df = raw_trials_metadata_df.copy()

        # Ensure numeric types for calculation
        df['step_index'] = df['step_index'].astype(int)
        min_step = df['step_index'].min()
        max_step = df['step_index'].max()

        # Calculate progression (0.0 to 1.0)
        df['norm_step'] = (df['step_index'] / (max_step - min_step)).round(2)

        is_src = df['step_index'] == min_step
        is_dst = df['step_index'] == max_step

        # Identify fixed points (anchors) vs transitions (morphs)
        df['stim_type'] = np.where(is_src | is_dst, 'anchor', 'morph')

        # Generate naming convention: Source_0.XX_Destination_0.YY
        partial_names = (
                df['src_cat'] + "_" +
                (1 - df['norm_step']).round(2).astype(str) + "_" +
                df['dst_cat'] + "_" +
                df['norm_step'].round(2).astype(str)
        )

        df['morph_name'] = np.where(is_src, df['src_cat'],
                                    np.where(is_dst, df['dst_cat'], partial_names))

        df['nearest_anchor'] = np.where((df['norm_step'] < 0.5), df['src_cat'], df['dst_cat'])

        # Clean up Anchor data (Anchors usually don't belong to a morphing pair)
        anchor_mask = df['stim_type'] == 'anchor'
        cols_to_null = ['pair_key', 'src_cat', 'dst_cat', 'step_index', 'norm_step']
        df.loc[anchor_mask, cols_to_null] = np.nan

        df.index = df['morph_name']
        df.rename_axis('morph', inplace=True)
        self.append(df)

    def append(self, df: pd.DataFrame) -> None:
        """Appends a new DataFrame and tracks its length for session splitting."""
        self._trial_lens.append(df.shape[0])
        self._metadata_df = pd.concat([self._metadata_df, df])

    def synchronize_with_data(self, combined_df: pd.DataFrame) -> None:
        """
        Filters metadata to match the indices present in an external DataFrame.
        Useful after dropping trials from neural/behavioral data.
        """
        metadata_lookup = self._metadata_df.drop_duplicates(subset='morph_name')
        self._metadata_df = metadata_lookup.reindex(combined_df.index).rename_axis(index='morph')
        self.disable_mask()
        self._update_masked_view()

    def get_morph_names(self, as_list: bool = False, ignore_mask: bool = False) -> Union[pd.Series, npt.NDArray]:
        """Returns the morph names, optionally as a list/array."""
        data = self.get_metadata(ignore_mask=ignore_mask)
        return data['morph_name'].values if as_list else data['morph_name']

    @property
    def morph_names(self):
        if self._use_mask:
            return self.masked_metadata['morph_name']
        else: return self.all_metadata['morph_name']

    @property
    def all_metadata(self):
        return self._metadata_df

    @property
    def masked_metadata(self):
        return self._masked_metadata


    def get_pair_keys(
            self,
            unique: bool = True,
            dropna: bool = True,
            values: bool = False
    ) -> Union[pd.Series, npt.NDArray]:
        """
        Retrieves trial pair keys.

        Args:
            unique: If True, returns only unique keys.
            dropna: If True, removes NaN values (common for anchors).
            values: If True, returns a numpy array instead of a Series.
        """
        pair_keys = self.get_metadata()['pair_key']

        if dropna:
            pair_keys = pair_keys.dropna()
        if unique:
            pair_keys = pd.Series(pair_keys.unique())

        return pair_keys.values if values else pair_keys

    @property
    def morph_steps(self) -> pd.DataFrame:
        df = self.get_metadata()['norm_step']
        return df

    def apply_mask(self, mask: np.ndarray) -> None:
        """Applies a general filter mask (e.g., performance or reaction time)."""
        self._filter_mask = mask
        self._update_masked_view()

    def apply_train_mask(self, mask: np.ndarray) -> None:
        """Applies a training/test split mask."""
        self._train_mask = mask
        self._update_masked_view()

    def _update_masked_view(self) -> None:
        """Internal helper to combine active masks."""
        combined_mask = None
        if self._filter_mask is not None:
            combined_mask = self._filter_mask
        if self._train_mask is not None:
            combined_mask = self._train_mask if combined_mask is None else (combined_mask & self._train_mask)

        if combined_mask is not None:
            self._masked_metadata = self._metadata_df[combined_mask]
            self._use_mask = True

    def disable_mask(self) -> None:
        """Clears all masks and returns to full dataset view."""
        self._filter_mask = None
        self._train_mask = None
        self._use_mask = False
        self._masked_metadata = None

    @property
    def anchor_mask(self) -> pd.Series:
        """Returns a boolean mask where trials are anchors."""
        return self.get_metadata()['stim_type'] == 'anchor'

    def get_metadata(self, ignore_mask: bool = False) -> pd.DataFrame:
        """Returns either the masked or full metadata DataFrame."""
        if self._use_mask and not ignore_mask:
            return self._masked_metadata
        return self._metadata_df

    def shuffle(self, random_state: int = 42) -> None:
        """Randomly shuffles the full metadata."""
        self._metadata_df = self._metadata_df.sample(frac=1, random_state=random_state)

    def reindex(self, sorted_idx: Union[List[int], np.ndarray], allow_mismatch: bool = False) -> None:
        """
        Sorts the current view (masked or full) based on provided indices.
        """
        target_df = self._masked_metadata if self._use_mask else self._metadata_df

        if (len(target_df) != len(sorted_idx)) and not allow_mismatch:
            raise ValueError(f"Index length ({len(sorted_idx)}) does not match data length ({len(target_df)})")

        if self._use_mask:
            self._masked_metadata = self._masked_metadata.iloc[sorted_idx]
        else:
            self._metadata_df = self._metadata_df.iloc[sorted_idx]

    def find_matching_pair_keys(self, search_term: str) -> Tuple[List[int], int]:
        """
        Finds integer indices where the pair_key contains the search_term string.
        """
        pair_keys = self.get_pair_keys(unique=False, dropna=False, values=True)
        matches = [i for i, pk in enumerate(pair_keys) if search_term in str(pk)]
        return matches, len(matches)

    def copy(self) -> 'TrialMetadata':
        """Returns a deep copy of the current object."""
        return copy.deepcopy(self)

    @property
    def nearest_anchor(self):
        return self.get_metadata()['nearest_anchor']

