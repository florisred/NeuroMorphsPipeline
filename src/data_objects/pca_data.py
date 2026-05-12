from data_objects.trial_metadata import TrialMetadata
import numpy.typing as npt
import pandas as pd
import numpy as np
import copy


class PCAData:
    def __init__(self, pca_type: str, pca_output: npt.NDArray, metadata: TrialMetadata, morph_names: pd.Index, explained_variance: npt.NDArray = None):
        self._pca_output = pca_output
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
        if self._normalized: return (self._pca_df - self._pca_df.mean()) / self._pca_df.std()
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

    def get_data_components(self, n_components: int):
        return self.pca_data.iloc[:, :n_components]

    def get_numeric_index(self):
        return np.arange(len(self.pca_data))

    def sort(self, train_test = ""):
        morph_names = list(self.metadata.morph_names)
        pair_keys =self.metadata.get_pair_keys()

        final_sequence = []
        for pair_key in pair_keys:
            matching_morphs = self.metadata_df[self.metadata.get_pair_keys(unique=False, dropna=False) == pair_key]
            src_name = f"{np.unique(matching_morphs['src_cat'])[0]}{train_test}"
            final_sequence.append(src_name)
            matching_morphs.sort_values(by='norm_step', inplace=True)
            for morph_name in matching_morphs.index:
                final_sequence.append(morph_name)
            if train_test != '':
                dst_name = f"{np.unique(matching_morphs['dst_cat'])[0]}_{train_test}"
            else: dst_name = f"{np.unique(matching_morphs['dst_cat'])[0]}"
            final_sequence.append(dst_name)

        name_to_orig_idx = {name: i for i, name in enumerate(morph_names)}
        new_indices = [name_to_orig_idx[name] for name in final_sequence if name in name_to_orig_idx]

        self._grouped_pca_data = self._grouped_pca_data.iloc[new_indices]
        self.metadata.reindex(new_indices, allow_mismatch=True)
