import logging
from typing import Optional, List
from data_objects.data_source import DataSource
from utils.utils import create_name_from_list
import pandas as pd
from sklearn.decomposition import PCA
from data_objects.trial_metadata import TrialMetadata
from data_objects.pca_data import PCAData
logger = logging.getLogger(__name__)


class PCAManager:
    """Handles the caching and generation of PCA data to keep the Analyzer clean."""

    def __init__(self, default_components: int = 3):
        self.default_components = default_components
        self.datasources: dict[str, DataSource] = {}
        self.cache: dict[str, any] = {}

    def clear_cache(self):
        self.cache.clear()

    def add_datasource(self, data_source: DataSource):
        key = data_source.data_type
        if key in self.datasources:
            logger.warning(f"Overwriting datasource: {key}")
        self.datasources[key] = data_source

    def prepare_data(self, pca_type: str, n_components: Optional[int] = None, subsets: Optional[List] = None):
        """Ensures the cache contains the requested pca_type for all datasources."""
        comps = n_components or self.default_components

        for key, ds in self.datasources.items():
            if pca_type == 'full':
                pca_name = f'{key}_full'
                if pca_name not in self.cache:
                    pca_data = self.run_pca(
                        all_data=ds.get_data(), fit_data=ds.get_anchors(),
                        n_components=comps, metadata=ds.get_metadata(), pca_type='full'
                    )
                    pca_data.set_name(pca_name)
                    self.cache[pca_name] = pca_data

            elif pca_type == 'subsets':
                subs = subsets or ds.find_stimulus_cycles(n=3)
                # Clear old subsets for this key
                keys_to_drop = [k for k in self.cache.keys() if 'subset' in k and k.startswith(key)]
                for k in keys_to_drop: self.cache.pop(k)

                for subset in subs:
                    pca_name = f'{key}_{create_name_from_list(subset)}'
                    if pca_name in self.cache: continue

                    temp_ds = ds.copy()
                    temp_ds.filter_transitions(subset)
                    pca_data = self.run_pca(
                        all_data=temp_ds.get_data(), n_components=comps,
                        fit_data=temp_ds.get_anchors(), metadata=temp_ds.get_metadata(), pca_type='subsets'
                    ) # ds for all data, temp_ds for subset data
                    pca_data.sort()
                    pca_data.set_name(pca_name)
                    self.cache[pca_name] = pca_data

    @staticmethod
    def run_pca(pca_type: str, metadata: TrialMetadata, all_data: pd.DataFrame, n_components: int, fit_data: pd.DataFrame=None):
        pca_model = PCA(n_components=n_components)
        if fit_data is not None:
            pca_model.fit(fit_data)
            pca_result = pca_model.transform(all_data)
        else:
            pca_result = pca_model.fit_transform(all_data)
        explained_variance = pca_model.explained_variance_ratio_
        pca_data = PCAData(
            pca_output = pca_result,
            explained_variance = explained_variance,
            metadata = metadata,
            pca_type = pca_type
        )
        return pca_data
