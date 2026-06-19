import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

from data_objects.data_source import DataSource
from analysis.pca_manager import PCAManager
from analysis.plots import distances, interactive, rdm_plot, subsets_plot, explained_variance, plot_3d  # Import our new standalone plots module
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
        '3d': plot_3d.create_3d_plot,
        'rdm': rdm_plot.rdm_analysis,
        'rdm_full': rdm_plot.rdm_analysis_full,
        'anchor_rdm': rdm_plot.anchor_rdm,
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
        'anchor_rdm': ['anchors'],
        'explained_variance_full': ['full'],
        'explained_variance_subsets': ['subsets'],
        '3d': ['full']
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
            plot_config: dict,

    ):
        subsets_n = plot_config.get('subsets_n', 3)
        n_components = plot_config.get('n_components', 'max')
        show = plot_config.get('show', False)
        subsets_with_variability = plot_config.get('subsets_with_variability', True)

        plot_config['output_dir'] = output_dir

        #params = {k: v for k, v in locals().items() if k != 'self'}

        output_dir.mkdir(parents=True, exist_ok=True)

        required_pca_types = {tuple(self.PCA_REQUIREMENTS[p]) for p in plot_types if p in self.PCA_REQUIREMENTS}
        for req_type in required_pca_types:
            logger.info(f'Loading PCA data for {req_type}')
            self.pca_manager.prepare_data(pca_types=req_type, n_components=n_components, subsets_n=subsets_n)

        for plot_type in plot_types:
            plot_func = self.PLOT_REGISTRY.get(plot_type)
            if plot_func:
                logger.info(f"Generating {plot_type} plots...")
                plot_func(self.pca_manager.cache, **plot_config)
