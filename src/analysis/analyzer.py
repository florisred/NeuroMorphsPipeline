import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

from analysis.pca_performer import PCAPerformer
from analysis.plot_creator import PlotCreator
from data_objects.data_source import DataSource
from utils.utils import create_name_from_list

# Setup basic logging to replace silent failures or prints
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

    def create_plots(self, plot_types: Union[List[str], Tuple[str]], output_dir: Path, triplets: Optional[List] = None):
        """Coordinates the creation of requested plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        needs_full_pca = any(p in plot_types for p in ['interactive', 'distances'])
        needs_triplet_pca = any(p in plot_types for p in ['triplets', 'rdm'])
        for ds_key in self._datasource_dict.keys():
            if needs_full_pca:
                self._ensure_pca(key=ds_key, pca_type='full')
            if needs_triplet_pca:
                self._ensure_pca(key=ds_key, pca_type='triplets', triplets=triplets)

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

        # Retrieve all calculated triplet keys for the next steps
        triplet_keys = [k for k in self._pca_dict.keys() if '_full' not in k]

        # 3. Triplet 2D Plots
        if 'triplets' in plot_types:
            for t_key in triplet_keys:
                self.plot_object.create_2d_plots(
                    pca_data=self._pca_dict[t_key],
                    output_dir=output_dir
                )

        # 4. RDM Analysis
        if 'rdm' in plot_types:
            # Group PCA data by their triplet name suffix
            # E.g., Groups 'TwoPhoton_triplet1' and 'Ephys_triplet1' together
            triplet_groups = {}
            for t_key in triplet_keys:
                # Extract the triplet name assuming the format is "{ds_key}_{triplet_name}"
                # We split on the first underscore to separate the ds_key from the rest
                ds_key = t_key.split('_')[0]
                triplet_name = t_key[len(ds_key) + 1:]

                if triplet_name not in triplet_groups:
                    triplet_groups[triplet_name] = []
                triplet_groups[triplet_name].append(self._pca_dict[t_key])

            # Run RDM for each unique triplet group
            for triplet_name, pca_data_list in triplet_groups.items():
                # Only run RDM if we have something to compare
                if len(pca_data_list) > 0:
                    self.plot_object.rdm_analysis(
                        pca_data_list=pca_data_list,
                        output_dir=output_dir
                    )

    def _ensure_pca(self, key: str, pca_type: str, triplets: Optional[List] = None):
        """Wrapper to check cache before running the heavy PCA computation."""
        if pca_type == 'full':
            full_key = f'{key}_full'
            if full_key not in self._pca_dict:
                self._prepare_pca(key=key, pca_type='full')

        elif pca_type == 'triplets':
            # We must run it to ensure all requested triplets are processed.
            # The _prepare_pca logic handles caching the individual triplet results.
            self._prepare_pca(key=key, pca_type='triplets', triplets=triplets)

    def _prepare_pca(self, key: str, pca_type: str = 'full', triplets: Optional[List] = None):
        """Runs PCA on the given data source and pca_type."""
        data_source = self._datasource_dict.get(key)
        if not data_source:
            raise ValueError(f"No data source found for key: {key}")

        if pca_type == 'full':
            pca_data = self.pca_performer.run_pca(
                all_data=data_source.get_data(),
                fit_data=data_source.get_anchors(),
                n_components=self.n_components,
                metadata=data_source.get_metadata(),
                pca_type=pca_type
            )
            pca_name = f'{key}_full'
            pca_data.set_name(pca_name)
            self._pca_dict[pca_name] = pca_data

        elif pca_type == 'triplets':
            if triplets is None:
                triplets = data_source.find_stimulus_cycles(n=3)

            for triplet in triplets:
                triplet_name = create_name_from_list(triplet)
                pca_name = f'{key}_{triplet_name}'

                # Check cache first to avoid re-running PCA for this specific triplet
                if pca_name in self._pca_dict:
                    continue

                temp_data_source = data_source.copy()
                temp_data_source.filter_transitions(triplet)

                pca_data = self.pca_performer.run_pca(
                    all_data=temp_data_source.get_data(),
                    n_components=self.n_components,
                    fit_data=temp_data_source.get_anchors(),
                    metadata=temp_data_source.get_metadata(),
                    pca_type=pca_type
                )
                pca_data.sort()
                pca_data.set_name(pca_name)
                self._pca_dict[pca_name] = pca_data
        else:
            raise ValueError(f"Unknown pca_type: {pca_type}")