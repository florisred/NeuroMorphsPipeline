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
        """Clears the PCA cache."""
        self.cache.clear()

    def add_datasource(self, data_source: DataSource):
        """
        Adds a datasource to the object for analysis.
        :param data_source: DataSource object to be added
        """
        key = data_source.get_data_type()
        if key in self.datasources:
            logger.warning(f"Overwriting datasource: {key}")
        self.datasources[key] = data_source

    def prepare_data(self, pca_types: tuple[str], n_components: Optional[int] = None, subsets: Optional[List] = None):
        """
        Ensures the cache contains the requested pca_type for all datasources.

        :param pca_types: type of pca used, used for identification later on
        :param n_components: number of PCA components
        :param subsets: list of subsets to use
        """
        comps = n_components or self.default_components

        # looks through every datasource that has been loaded
        for key, ds in self.datasources.items():
            # if the pca_type is full, make sure that it has that
            if 'full' in pca_types and not ds.is_split:
                pca_name = f'{key}_full'
                if pca_name not in self.cache:
                    pca_data = self.run_pca(
                        all_data=ds.get_data(), fit_data=ds.get_anchors(),
                        n_components=comps, metadata=ds.get_metadata(), pca_type='full'
                    ) # fits on the anchors only.
                    pca_data.set_name(pca_name)
                    self.cache[pca_name] = pca_data
            if 'split_full' in pca_types and ds.is_split:
                pca_name = f'{key}_full'
                if pca_name not in self.cache:
                    test_ds = ds.copy()
                    test_ds.train_test_mask('test')
                    train_ds = ds.copy()
                    train_ds.train_test_mask('train')
                    pca_data = self.run_pca(
                        all_data=test_ds.get_data(),
                        fit_data=train_ds.get_anchors(),
                        n_components=comps,
                        metadata=test_ds.get_metadata(),
                        pca_type='split_full',
                        train_test = '_test'
                    )
                    pca_data.set_name(pca_name)
                    self.cache[pca_name] = pca_data

            if 'subsets' in pca_types and not ds.is_split:
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
                    pca_data.set_name(pca_name)
                    self.cache[pca_name] = pca_data

    @staticmethod
    def run_pca(pca_type: str, metadata: TrialMetadata, all_data: pd.DataFrame, n_components: int, fit_data: pd.DataFrame=None, train_test= ''):
        """
        :param pca_type: type of pca used, used for identification later on
        :param metadata: TrialMetadata object
        :param all_data: dataframe of all data to be transformed in the PCA model (will also be used to fit if no fit_data is provided)
        :param n_components: number of PCA components
        :param fit_data: dataframe of all data used to fit the pca model
        :return: PCAData object, with the TrialMetadata Object and the transformed data
        """
        pca_model = PCA(n_components=n_components)
        if fit_data is not None: # if fit_data is none, just fit the data on all the data!
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
        pca_data.sort(train_test)
        return pca_data
