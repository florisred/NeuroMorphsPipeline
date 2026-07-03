import logging
from typing import Optional, List
from data_objects.data_source import DataSource
from utils.utils import create_name_from_list
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
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
        key = data_source.data_type
        if key in self.datasources:
            logger.warning(f"Overwriting datasource: {key}")
        self.datasources[key] = data_source

    def prepare_data(self, pca_types: tuple[str], n_components: Optional[int] = None, subsets_n: Optional[int] = None, pca_on_anchors: bool = False):
        """
        Ensures the cache contains the requested pca_type for all datasources.

        :param pca_types: type of pca used, used for identification later on
        :param n_components: number of PCA components
        :param subsets_n: number of subsets to use -> 3 for triplets
        """
        comps = n_components or self.default_components

        # looks through every datasource that has been loaded
        for key, ds in self.datasources.items():
            # if the pca_type is full, make sure that it has that
            if 'full' in pca_types:

                if pca_name not in self.cache:
                    if pca_on_anchors:
                        fit_data = ds.anchors
                        pca_name = f'{key}AnchorSpace_full'
                    else:
                        fit_data = ds.data
                        pca_name = f'{key}_full'
                    pca_data = self.run_pca(
                        all_data=ds.data, fit_data=fit_data,
                        n_components=comps, metadata=ds.metadata, pca_type='full'
                    ) # fits on the anchors only.
                    pca_data.set_name(pca_name)
                    self.cache[pca_name] = pca_data


            if 'anchors' in pca_types:
                pca_name = f'{key}_anchors'
                if pca_name not in self.cache:
                    metadata=ds.metadata.copy()
                    metadata.apply_mask(metadata.anchor_mask)
                    pca_data = self.run_pca(
                        all_data=ds.anchors, fit_data=ds.anchors,
                        n_components=comps, metadata=metadata, pca_type='anchors', sort=False
                    )
                    pca_data.set_name(pca_name)
                    self.cache[pca_name] = pca_data


            if 'subsets' in pca_types:
                subs = ds.find_stimulus_cycles(n=subsets_n)
                # Clear old subsets for this key
                keys_to_drop = [k for k in self.cache.keys() if 'subset' in k and k.startswith(key)]
                for k in keys_to_drop: self.cache.pop(k)

                for subset in subs:
                    temp_ds = ds.copy()
                    temp_ds.filter_transitions(subset)
                    if pca_on_anchors:
                        fit_data = temp_ds.anchors
                        pca_name = f'{key}AnchorSpace_{create_name_from_list(subset)}'
                    else:
                        fit_data = temp_ds.data
                        pca_name = f'{key}_{create_name_from_list(subset)}'
                    if pca_name in self.cache: continue
                    logger.info(f"Creating PCA for subset {key}AnchorSpace - {pca_name}")
                    pca_data = self.run_pca(
                        all_data=temp_ds.data, n_components=comps,
                        fit_data=fit_data, metadata=temp_ds.metadata, pca_type='subsets'
                    ) # ds for all data, temp_ds for subset data
                    pca_data.set_name(pca_name)
                    pca_data.sort_subsets(subset)
                    self.cache[pca_name] = pca_data


    @staticmethod
    def run_pca(pca_type: str, metadata: TrialMetadata, all_data: pd.DataFrame, n_components: int, fit_data: pd.DataFrame=None):
        """
        :param pca_type: type of pca used, used for identification later on
        :param metadata: TrialMetadata object
        :param all_data: dataframe of all data to be transformed in the PCA model (will also be used to fit if no fit_data is provided)
        :param n_components: number of PCA components
        :param fit_data: dataframe of all data used to fit the pca model
        :return: PCAData object, with the TrialMetadata Object and the transformed data
        """

        if n_components == 'max':
            if fit_data is not None: n_components=min(fit_data.shape)
            else: n_components=min(all_data.shape)
        pca_model = PCA(n_components=n_components)
        if fit_data is not None: # if fit_data is none, just fit the data on all the data!
            if min(fit_data.shape) < n_components:
                pca_model = PCA(n_components=min(fit_data.shape))
                logger.warning(f"PCA fitting with {n_components} PCA components is not possible. Using {min(fit_data.shape)} instead.")
            pca_model.fit(fit_data)
            pca_result = pca_model.transform(all_data)
        else:
            if min(all_data.shape) < n_components:
                pca_model = PCA(n_components=min(all_data.shape))
                logger.warning(
                    f"PCA fitting with {n_components} PCA components is not possible. Using {min(all_data.shape)} instead.")
            pca_result = pca_model.fit_transform(all_data)
        explained_variance = pca_model.explained_variance_ratio_
        pca_data = PCAData(
            pca_output = pca_result,
            explained_variance = explained_variance,
            metadata = metadata,
            pca_type = pca_type,
            morph_names = all_data.index,
            raw_data = all_data
        )
        pca_data.metadata.synchronize_with_data(combined_df=pca_data.pca_data)
        pca_data.sort()
        return pca_data
