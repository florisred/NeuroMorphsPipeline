import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

from analysis.pca_performer import PCAPerformer
from analysis.plot_creator import PlotCreator
from data_objects.data_source import DataSource
from utils.utils import create_name_from_list

# Set up basic logging to replace silent failures or prints
logger = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self._datasource_dict: dict[str, DataSource] = {}
        self._pca_dict: dict[str, any] = {}  # Cache for PCA results

        self.plot_object = PlotCreator()
        self.pca_performer = PCAPerformer()

    def load_datasource(self, data_source: DataSource):
        """Loads a datasource into the analyzer."""
        key = data_source.data_type
        if key in self._datasource_dict:
            logger.warning(f"Overwriting existing datasource with key: {key}")
        self._datasource_dict[key] = data_source

    def create_plots(self, plot_types: Union[List[str], Tuple[str]], output_dir: Path, subsets: Optional[List] = None, n_components: Optional[int] = None, avg_only: Optional[bool] = False):
        """Coordinates the creation of requested plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        needs_full_pca = any(p in plot_types for p in ['interactive', 'distances'])
        needs_subset_pca = any(p in plot_types for p in ['subsets', 'rdm'])
        
        if n_components is None:
            n_components = self.n_components

        for ds_key in self._datasource_dict.keys():
            if needs_full_pca:
                self._ensure_pca(key=ds_key, pca_type='full', n_components=n_components)
            if needs_subset_pca:
                self._ensure_pca(key=ds_key, pca_type='subsets', subsets=subsets, n_components=n_components)

        if 'interactive' in plot_types:
            for ds_key in self._datasource_dict.keys():
                full_key = f'{ds_key}_full'
                self.plot_object.create_interactive_plot(
                    pca_data=self._pca_dict[full_key],
                    output_dir=output_dir
                )

        if 'distances' in plot_types:
            for ds_key in self._datasource_dict.keys():
                full_key = f'{ds_key}_full'
                self.plot_object.calculate_distances(
                    pca_data=self._pca_dict[full_key],
                    output_dir=output_dir
                )
        if 'subsets' in plot_types:
            self.plot_object.create_2d_plots(
                pca_data_dict=self._pca_dict,
                output_dir=output_dir
            )

        # 4. RDM Analysis
        if 'rdm' in plot_types:
            self.plot_object.rdm_analysis(
                pca_data_dict=self._pca_dict,
                output_dir=output_dir,
                avg_only=avg_only,
                n_components=n_components
            )
    
    def _ensure_pca(self, key: str, pca_type: str, subsets: Optional[List] = None, n_components: Optional[int] = None):
        """Wrapper to check cache before running the heavy PCA computation."""
        if pca_type == 'full':
            full_key = f'{key}_full'
            if full_key not in self._pca_dict:
                self._prepare_pca(key=key, pca_type='full')

        elif pca_type == 'subsets':
            self._prepare_pca(key=key, pca_type='subsets', subsets=subsets, n_components=n_components)


    def _prepare_pca(self, key: str, pca_type: str = 'full', subsets: Optional[List] = None, n_components: Optional[int] = None):
        """Runs PCA on the given data source and pca_type."""
        data_source = self._datasource_dict.get(key)
        if not data_source:
            raise ValueError(f"No data source found for key: {key}")

        if pca_type == 'full':
            pca_data = self.pca_performer.run_pca(
                all_data=data_source.get_data(),
                fit_data=data_source.get_anchors(),
                n_components=n_components,
                metadata=data_source.get_metadata(),
                pca_type=pca_type
            )
            pca_name = f'{key}_full'
            pca_data.set_name(pca_name)
            self._pca_dict[pca_name] = pca_data

        elif pca_type == 'subsets':
            if subsets is None:
                subsets = data_source.find_stimulus_cycles(n=3)

            for subset in subsets:
                subset_name = create_name_from_list(subset)
                pca_name = f'{key}_{subset_name}'

                # Check the cache first to avoid re-running PCA for this specific subset
                if pca_name in self._pca_dict:
                    continue

                temp_data_source = data_source.copy()
                fit_data = temp_data_source.get_anchors()
                temp_data_source.filter_transitions(subset)

                pca_data = self.pca_performer.run_pca(
                    all_data=temp_data_source.get_data(),
                    n_components=n_components,
                    fit_data=fit_data,
                    metadata=temp_data_source.get_metadata(),
                    pca_type=pca_type
                )
                pca_data.sort()
                pca_data.set_name(pca_name)
                self._pca_dict[pca_name] = pca_data

        else:
            raise ValueError(f"Unknown pca_type: {pca_type}")
        
    
