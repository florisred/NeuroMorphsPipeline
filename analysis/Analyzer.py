from MixIns.PCAMixin import PCAMixin
from MixIns.PlotMixIns import *
from data_loader.DataSource import DataSource
from utils.utils import create_name_from_list



class Analyzer(PCAMixin, Plot2DMixIn, PlotInteractiveMixIn, PlotDistancesMixIn):#, Plot3DMixIn, ):
    def __init__(self):
        self.n_components = 3
        self._datasource_dict ={}
        self._pca_dict = {}

    def load_datasource(self, data_source: DataSource):
        key = data_source.data_type
        self._datasource_dict[key] = data_source

    def _prepare_pca(self, data_source: DataSource=None, key=None, pca_type: str = 'full', triplets = None):
        """
        runs PCA on the given data source and pca_type
        """
        if not (data_source is None) ^ (key is None):
            raise Exception('either data_source or key is required, not both')
        if key is not None:
            data_source = self._datasource_dict.get(key)
        key = data_source.data_type
        match pca_type:
            case 'full':
                data = data_source.get_data()
                anchors = data_source.get_anchors()
                pca_data =  self._run_pca(
                    all_data = data,
                    fit_data=anchors,
                    n_components = self.n_components,
                    metadata=data_source.get_metadata(),
                    pca_type=pca_type
                )
                pca_name = f'{key}_full'
                pca_data.set_name(pca_name)
                self._pca_dict[pca_name] = pca_data
            case 'triplets':
                if triplets is None:
                    triplets = data_source.find_stimulus_cycles(n=3)
                for triplet in triplets:
                    temp_data_source = data_source.copy()
                    triplet_name = create_name_from_list(triplet)
                    temp_data_source.filter_transitions(triplet)
                    data = temp_data_source.get_data()
                    anchors = temp_data_source.get_anchors()
                    pca_data = self._run_pca(
                        all_data = data,
                        n_components = self.n_components,
                        fit_data = anchors,
                        metadata=temp_data_source.get_metadata(),
                        pca_type=pca_type
                    )
                    pca_data.sort()
                    pca_name = f'{key}_{triplet_name}'
                    pca_data.set_name(pca_name)
                    self._pca_dict[pca_name] = pca_data





    def create_plots(self, plot_types: list|tuple, output_dir: Path, triplets: list = None):
        if 'interactive' in plot_types:
            for datasource_key in self._datasource_dict.keys():
                full_key = f'{datasource_key}_full'
                if full_key not in self._pca_dict.keys():
                    self._prepare_pca(key = datasource_key, pca_type = 'full')
                self._create_interactive_plot(
                    pca_data = self._pca_dict[full_key],
                    output_dir = output_dir
                )
        if 'triplets' in plot_types:
            for datasource_key in self._datasource_dict.keys():
                self._prepare_pca(key = datasource_key, pca_type = 'triplets', triplets=triplets)
            triplet_keys = [key for key in self._pca_dict.keys() if 'triplet' in key]
            for key in triplet_keys:
                self.create_2d_plots(
                    pca_data = self._pca_dict[key],
                    output_dir = output_dir
                )
        if 'distances' in plot_types:
            for datasource_key in self._datasource_dict.keys():
                full_key = f'{datasource_key}_full'
                if full_key not in self._pca_dict.keys():
                    self._prepare_pca(key=datasource_key, pca_type='full')
                self.calculate_distances(
                    pca_data=self._pca_dict[full_key],
                    output_dir = output_dir
                )



