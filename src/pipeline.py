from pathlib import Path
from warnings import warn
from analysis.plots.ori_ring_plot import ori_explained_variance_plot
from data_objects.ori_data_source import OriTwoPhotonDataSource, OriPixeLWiseDataSource, OriGaborDataSource, \
    OriDistGaborDataSource
from scripts.generate_config import generate_config
import json
from data_objects.two_photon_data_source import TwoPhotonDataSource
from analysis.analyzer import Analyzer
from data_objects.stimulus_data_source import StimulusGaborDataSource, StimulusPixelWiseDataSource, DistributedGaborDataSource
from src.analysis.plots.rdm_plot_new import rdm_analysis_ori
from src.analysis.plots.ori_ring_plot import ori_ring_plot

class Pipeline:
    """
    Class that handles the data pipeline
    """

    def __init__(self, data_folder:str|None = None):

        # set preliminary vars
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "config"
        settings_file = config_dir / "settings.json"

        self._ori_data_sources = {}

        # check if there is a config file, if not let user generate it
        if not settings_file.is_file():
            warn("Configuration file not found.")
            if input("Make config file from defaults? (y/n)").lower() == "y":
                generate_config(
                    config_dir = config_dir,
                    settings_file = settings_file
                )
            else:
                print("Aborting")
                exit()

        # load the settings
        with open(settings_file, "r") as f:
            self.settings = json.load(f)

        if data_folder is not None:
            self.data_dir = Path(data_folder)
        else:
            self.data_dir = Path(self.settings["DATA_FOLDER"])
        self.session_dirs = [dire for dire in self.data_dir.iterdir() if dire.name not in ["output", "stimuli", 'rfsizes'] and not dire.name.startswith(".")]
        self.output_dir = self.data_dir / "output" / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = Analyzer()




    def load_data_sources(
            self,
            sources: list,
            n_repeat_trials:int=7,
            gabornet_params: dict|None=None,
            h5_locations_textures:dict|None=None,
            h5_locations_ori:dict|None=None
    ):
        if ('2p' in sources or 'GaborFilterBank' in sources or 'GaborNet' in sources or 'PixelWise' in sources) and ('ori_2p' in sources or 'ori_GaborFilterBank' in sources or 'ori_GaborNet' in sources or 'ori_PixelWise' in sources):
            raise ValueError("Due to current limitations, it is not possible to load both orientation data and texture data at the same time. Please perform these analyses separately.")

        if '2p' in sources:
            if h5_locations_textures is None: raise ValueError("H5 locations for textures not provided")
            two_photon = TwoPhotonDataSource(
                file_paths=self.session_dirs,
                data_location=h5_locations_textures['data'],
                metadata_locations = h5_locations_textures['labels']
            )
            two_photon.load_data()
            self.analyzer.load_datasource(data_source=two_photon)

        if 'GaborFilterBank' in sources:
            gabor_bank = StimulusGaborDataSource(
                file_paths=[self.data_dir / 'stimuli'],
                gabor_params=self.settings["gabor_params"],
                output_dir=self.data_dir / 'output',
            )
            gabor_bank.load_data()
            self.analyzer.load_datasource(gabor_bank)

        if 'GaborNet' in sources:
            gabor_net = DistributedGaborDataSource(
                file_paths=[self.data_dir / 'stimuli'],
                rf_dstr_path=self.data_dir / 'rfsizes',
                gabor_params=gabornet_params,
                output_dir=self.data_dir / 'output',
            )
            gabor_net.load_data(rf_size_multiplier=1)
            self.analyzer.load_datasource(gabor_net)

        if 'RetinoDivNormGaborNet' in sources:
            retinodivnorm_gabor_net = DistributedGaborDataSource(
                file_paths=[self.data_dir / 'stimuli'],
                rf_dstr_path=self.data_dir / 'rfsizes',
                gabor_params=gabornet_params,
                output_dir=self.data_dir / 'output',
                retinodivnorm=True
            )
            retinodivnorm_gabor_net.load_data(rf_size_multiplier=1)
            self.analyzer.load_datasource(
                data_source=retinodivnorm_gabor_net,
            )


        if 'PixelWise' in sources:
            stimulus_pixel = StimulusPixelWiseDataSource(
                file_paths=[self.data_dir / 'stimuli'],
            )
            stimulus_pixel.load_data()
            self.analyzer.load_datasource(stimulus_pixel)

        if 'ori_2p' in sources:
            if h5_locations_ori is None: raise ValueError("H5 locations for orientation data not provided")
            ori_two_photon = OriTwoPhotonDataSource(
                file_paths=self.session_dirs,
                data_location=h5_locations_ori['data'],
                metadata_locations=h5_locations_ori['labels'],
            )
            ori_two_photon.load_data()
            self._ori_data_sources['Neural-Sate Space'] = ori_two_photon
        if 'ori_GaborFilterBank' in sources:
            ori_gabor = OriGaborDataSource(
                file_paths=[self.data_dir / 'stimuli'],
                output_dir=self.output_dir / 'output',
                gabor_params=self.settings["gabor_params"]
            )
            ori_gabor.load_data()
            self._ori_data_sources['Gabor Filter Bank Space'] = ori_gabor
        if 'ori_GaborNet' in sources:
            ori_dist_gabor = OriDistGaborDataSource(
                file_paths=[self.data_dir / 'stimuli'],
                output_dir=self.output_dir / 'output',
                rf_dstr_path=self.data_dir / 'rfsizes',
                gabor_params=gabornet_params
            )
            ori_dist_gabor.load_data()
            self._ori_data_sources['GaborNet State Space'] = ori_dist_gabor
        if 'ori_RetinodivnormGaborNet' in sources:
            ori_retinodivnorm = OriDistGaborDataSource(
                file_paths=[self.data_dir / 'stimuli'],
                output_dir=self.output_dir / 'output',
                rf_dstr_path=self.data_dir / 'rfsizes',
                gabor_params=gabornet_params,
                retinodivnorm=True
            )
            ori_retinodivnorm.load_data()
            self._ori_data_sources['RetinodivnormGaborNet State Space'] = ori_retinodivnorm
        if 'ori_PixelWise' in sources:
            ori_pixel = OriPixeLWiseDataSource(
                file_paths=[self.data_dir / 'stimuli'],
                output_dir=self.output_dir / 'output'
            )
            ori_pixel.load_data()
            self._ori_data_sources['Raw Pixel State-Space'] = ori_pixel
        if 'ori_pca' in sources:
            n_components=3
            ori_output_dir = self.output_dir / 'ori_plots'
            ori_output_dir.mkdir(exist_ok=True)

            pca_data_dict = {}
            for key, ori_data_source in self._ori_data_sources.items():
                ori_data_source.pca_data(n_components=n_components, return_data=False)
                pca_data = ori_data_source.pca_data_object
                pca_data.set_name(key)
                pca_data_dict[key] = pca_data
            kwargs = {}
            kwargs['output_dir'] = Path(
                f'{self.data_dir}/output/plots/ori_plots/')
            kwargs['show'] = True
            rdm_analysis_ori(pca_data_dict, **kwargs)

            for key, ori_data_source in self._ori_data_sources.items():
                pca_data = ori_data_source.pca_data(n_components=n_components)
                explained_variance = ori_data_source.explained_variance
                ori_ring_plot(pca_data, title=key, output_dir=ori_output_dir)
                ori_explained_variance_plot(explained_variance, title=key, output_dir=ori_output_dir)


    def create_plots(self, plot_types: list, plot_config: dict):
        plot_config['output_dir'] = self.output_dir
        self.analyzer.create_plots(plot_types=plot_types, plot_config=plot_config)

    def classify(self,):
        self.analyzer.classify()

