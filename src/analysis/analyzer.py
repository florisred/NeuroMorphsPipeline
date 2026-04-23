import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

from data_objects.data_source import DataSource
from analysis.pca_manager import PCAManager
from analysis.plots import distances, interactive, rdm_plot, subsets_plot, explained_variance  # Import our new standalone plots module
from analysis.classification.classify import  classify as classify_data

logger = logging.getLogger(__name__)


class Analyzer:
    """
    Class that handles the coordination of PCA analyses and Plot creation
    """
    # Which plot type is which function
    PLOT_REGISTRY = {
        'interactive': interactive.create_interactive_plot,
        'distances': distances.calculate_distances,
        'subsets': subsets_plot.create_subset_plots,
        'rdm': rdm_plot.rdm_analysis,
        'rdm_full': rdm_plot.rdm_analysis_full,
        'rdm_split_full': rdm_plot.rdm_analysis_full,
        'distances_split': distances.calculate_distances,
        'classification': classify_data,
        'explained_variance_full': explained_variance.plot_variance,
        'explained_variance_subsets': explained_variance.plot_variance
    }
    # what type of PCA each plot type requires
    PCA_REQUIREMENTS = {
        'interactive': ['full'],
        'distances': ['pairs'],
        'subsets': ['subsets'],
        'rdm': ['subsets'],
        'rdm_full': ['full'],
        'rdm_split_full': ['full', 'split_full'],
        'distances_split': ['split_full'],
        'classification': ['split_full'],
        'explained_variance_full': ['full'],
        'explained_variance_subsets': ['subsets'],
    }

    def __init__(self, n_components: int = 3):
        self.pca_manager = PCAManager(default_components=n_components)

    def load_datasource(self, data_source: DataSource):
        self.pca_manager.add_datasource(data_source)

    def classify(self):
        classify_data(self.pca_manager.datasources)

    def create_plots(
            self,
            plot_types: Union[List[str], Tuple[str]],
            output_dir: Path,
            subsets: Optional[List] = None,
            n_components: Optional[int] = None,
            avg_only: bool = False,
            remove_prev: bool = True,
            show: bool = False,
            components_to_use: list[int] = None,
            with_variability: bool = False,
    ):
        params = {k: v for k, v in locals().items() if k != 'self'}

        if remove_prev:
            self.pca_manager.clear_cache()
        output_dir.mkdir(parents=True, exist_ok=True)

        required_pca_types = {tuple(self.PCA_REQUIREMENTS[p]) for p in plot_types if p in self.PCA_REQUIREMENTS}
        for req_type in required_pca_types:
            # req_type will now be a tuple like ('full',)
            self.pca_manager.prepare_data(pca_types=req_type, n_components=n_components, subsets=subsets)

        for plot_type in plot_types:
            plot_func = self.PLOT_REGISTRY.get(plot_type)
            if plot_func:
                logger.info(f"Generating {plot_type} plots...")
                plot_func(self.pca_manager.cache, **params)
