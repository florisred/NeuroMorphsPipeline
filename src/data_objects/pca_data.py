from data_objects.trial_metadata import TrialMetadata
import numpy.typing as npt
import pandas as pd
import numpy as np
import copy

class PCAData:
    def __init__(self, pca_type: str, pca_output: npt.NDArray, metadata: TrialMetadata, morph_names: pd.Index, raw_data: pd.DataFrame|None, explained_variance: npt.NDArray = None):
        self._pca_output = pca_output
        self.raw_data = raw_data
        self.metadata = metadata
        self.explained_variance = explained_variance
        self._pca_df = pd.DataFrame(pca_output, index=metadata.morph_names, columns = [f'Component{i+1}' for i in range(pca_output.shape[1])])
        self._pca_type = pca_type
        self._pca_name = 'pca'
        self._grouped_pca_data = pd.DataFrame(pca_output, index=morph_names).groupby('morph_name').mean()
        self._normalized = False


    @property
    def normalized(self):
        return self._normalized

    def normalize(self, inplace=False):
        if inplace:
            self._normalized = True
            return None
        else:
            cp = self.copy()
            cp._normalized = True
            return cp

    @property
    def n_unique_anchors(self):
        return len(np.unique(self.anchors))

    def copy(self):
        return copy.deepcopy(self)

    def set_name(self, name: str):
        self._pca_name = name

    @property
    def trial_data(self) -> pd.DataFrame:
        if self._normalized:
            pca_df = self._pca_df.copy()
            mean_grouped_data = self._grouped_pca_data.mean()
            mean_grouped_data.index = pca_df.columns
            std = self._grouped_pca_data.std()
            std.index = pca_df.columns

            norm = (pca_df - mean_grouped_data) / std
            return norm

        else: return self._pca_df

    @property
    def data_source(self):
        try: return self.name.split('_')[0]
        except: return 'unknown'


    @property
    def anchors(self) -> pd.DataFrame:
        """
        Retrieves the data of all the anchors and returns it
        :return: a DataFrame of the anchor data
        """
        return self.pca_data[self.metadata.anchor_mask]

    @property
    def name(self):
        return self._pca_name
    @property
    def metadata_df(self) -> pd.DataFrame:
        return self.metadata.get_metadata()
    @property
    def pca_data(self) -> pd.DataFrame:
        if self._normalized: return (self._grouped_pca_data - self._grouped_pca_data.mean()) / self._grouped_pca_data.std()
        return self._grouped_pca_data
    @property
    def pca_type(self) -> str:
        return self._pca_type

    def mean(self):
        self_copy = self.copy()
        normalized = self_copy._normalized
        self_copy._normalized = False
        self_copy._pca_df = self_copy._pca_df.groupby(by=self_copy._pca_df.index).mean()
        self_copy._normalized = normalized
        self_copy.metadata.synchronize_with_data(self_copy.pca_data)

        return self_copy

    def get_data_components(self, n_components: int):
        if n_components == 'max':
            return self.pca_data
        return self.pca_data.iloc[:, :n_components]

    def get_numeric_index(self):
        return np.arange(len(self.pca_data))

    def sort(self):
        morph_names = list(self.metadata.morph_names)
        pair_keys =self.metadata.get_pair_keys()

        final_sequence = []
        for pair_key in pair_keys:
            matching_morphs = self.metadata_df[self.metadata.get_pair_keys(unique=False, dropna=False) == pair_key]
            src_name = f"{np.unique(matching_morphs['src_cat'])[0]}"
            final_sequence.append(src_name)
            matching_morphs.sort_values(by='norm_step', inplace=True)
            for morph_name in matching_morphs.index:
                final_sequence.append(morph_name)
            else: dst_name = f"{np.unique(matching_morphs['dst_cat'])[0]}"
            final_sequence.append(dst_name)

        name_to_orig_idx = {name: i for i, name in enumerate(morph_names)}
        new_indices = [name_to_orig_idx[name] for name in final_sequence if name in name_to_orig_idx]

        self._grouped_pca_data = self._grouped_pca_data.iloc[new_indices]
        self.metadata.reindex(new_indices, allow_mismatch=True)


    def sort_subsets(self, subset:list):
        anchor_list = []
        for pair_key in subset:
            for anchs in pair_key.split('__'):
                anchor_list.append(anchs)
        anchor_array = np.array(anchor_list)
        anchors = np.unique(anchor_array)
        anchors = np.append(anchors, anchors[0])
        morph_names = np.unique(self.metadata.morph_names.to_numpy())
        final_sequence = [anchors[0]]
        for end_anchor in anchors[1:]:
            begin_anchor = final_sequence[-1]
            valid_morphs = []
            for morph_name in morph_names:
                if begin_anchor in morph_name and end_anchor in morph_name:
                    valid_morphs.append(morph_name)

            split_name = valid_morphs[0].split('_')
            begin_anchor_idx = split_name.index(begin_anchor)
            dist_from_begin_anchor = 1-float(split_name[begin_anchor_idx+1])
            if dist_from_begin_anchor >0.5:
                valid_morphs.reverse()
            for morph_name in valid_morphs:
                final_sequence.append(morph_name)
            final_sequence.append(end_anchor)
        final_sequence.pop(-1)
        final_sequence = np.array(final_sequence)

        all_morph_names = np.array(self.metadata.get_morph_names(as_list=True))
        name_to_long_idx = {}
        for i, name in enumerate(all_morph_names):
            if name not in name_to_long_idx:
                name_to_long_idx[name] = i  # keep first occurrence only
        reorder_indices = [name_to_long_idx[name] for name in final_sequence]
        self.reindex(reorder_indices)


    def reindex(self, reorder_indices: list):
        self._grouped_pca_data = self._grouped_pca_data.iloc[reorder_indices]
        self.metadata.reindex(reorder_indices, allow_mismatch=True)

