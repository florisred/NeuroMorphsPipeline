from data_objects.trial_metadata import TrialMetadata
import numpy.typing as npt
import pandas as pd
import numpy as np
import re

class PCAData:
    def __init__(self, pca_type: str, pca_output: npt.NDArray, metadata: TrialMetadata):
        self._pca_output = pca_output
        self.metadata = metadata
        self._pca_df = pd.DataFrame(pca_output, index=metadata.get_morph_names(), columns = [f'Component{i+1}' for i in range(pca_output.shape[1])])
        self._pca_type = pca_type
        self._pca_name = 'pca'

    def set_name(self, name: str):
        self._pca_name = name

    @property
    def name(self):
        return self._pca_name
    @property
    def metadata_df(self) -> pd.DataFrame:
        return self.metadata.get_metadata()
    @property
    def pca_data(self) -> pd.DataFrame:
        return self._pca_df
    @property
    def pca_type(self) -> str:
        return self._pca_type

    def get_data_components(self, n_components: int):
        return self._pca_df.iloc[:, :n_components]

    def get_numeric_index(self):
        return np.arange(len(self.pca_data))

    def sort(self):
        """
        Returns a list of integer indices that sorts the input
        according to the cyclical material path.
        """
        morph_names = self.metadata.get_morph_names(as_list=True)
        anchor_order = sorted(self.metadata.get_anchor_names())

        path = []
        for i in range(len(anchor_order)):
            start = anchor_order[i]
            end = anchor_order[(i + 1) % len(anchor_order)]
            path.append((start, end))

        sorted_names = []
        for start, end in path:
            # 1. Add the anchor
            if start in morph_names:
                sorted_names.append(start)

            # 2. Find and sort the transition leg
            leg = []
            for m in morph_names:
                if start in m and end in m and start != end:
                    match = re.search(rf"{start}_([\d.]+)", m)
                    if match:
                        weight = float(match.group(1))
                        leg.append((m, weight))

            leg.sort(key=lambda x: x[1], reverse=True)
            sorted_names.extend([m[0] for m in leg])

        # --- THE INTEGRATION STEP ---
        # Create a mapping of {name: desired_position}
        name_to_pos = {name: i for i, name in enumerate(sorted_names)}

        # Generate the index array based on where each current label should go
        # This handles duplicates if your 'morph_names' has multiple rows per morph
        sorted_idx = sorted(range(len(morph_names)),
                            key=lambda k: name_to_pos.get(morph_names[k], 999))

        self._pca_df = self._pca_df.iloc[sorted_idx]
        self.metadata.sort(sorted_idx)

