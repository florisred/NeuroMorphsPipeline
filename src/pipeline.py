from pathlib import Path
from warnings import warn

from optype import DoesStr

from scripts.generate_config import generate_config
import json
from data_objects.two_photon_data_source import TwoPhotonDataSource
from analysis.analyzer import Analyzer
from data_objects.stimulus_data_source import StimulusGaborDataSource, StimulusPixelWiseDataSource, DistributedGaborDataSource

class Pipeline:
    """
    Class that handles the data pipeline
    """

    def __init__(self):

        # set preliminary vars
        self.two_photon_pairs = None
        self.two_photon_triplets = None
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "config"
        settings_file = config_dir / "settings.json"

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


        self.data_dir = Path(self.settings["DATA_FOLDER"])
        self.session_dirs = [dire for dire in self.data_dir.iterdir() if dire.name not in ["output", "stimuli", 'rfsizes'] and not dire.name.startswith(".")]
        self.output_dir = self.data_dir / "output" / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = Analyzer()


    def load_two_photon(self, split:bool = False, train_percent = 0.7):
        """
        Loads the two photon data into a DataSource Object, retrieves the location from the config file
        :param split: Boolean to indicate whether to split the data into train and test categories
        """
        self.test()

        two_photon = TwoPhotonDataSource(
            file_paths=self.session_dirs,
            data_location=self.settings["PSEUDOPOP_DATA"]
        )
        two_photon.load_data(split=split, train_percent=train_percent)
        self.two_photon_triplets = two_photon.find_stimulus_cycles(n=3) # finds possible triplets for the graphs later on
        two_photon_pairs = two_photon.metadata.get_pair_keys(unique=True, dropna=True) # finds all pairs for the graphs
        self.two_photon_pairs = [[pair] for pair in two_photon_pairs]
        self.analyzer.load_datasource(data_source=two_photon) # loads the datasource into the Analyzer() object

    def load_stimuli(self, n_neurons=500):
        """
        Loads the stimulus data into a DataSource Object, retrieves the location from the config file
        """
        self.test()
        distributed_gabor = DistributedGaborDataSource(
            file_paths=[self.data_dir / 'stimuli'],
            rf_dstr_path = self.data_dir / 'rfsizes',
            gabor_params = self.settings["gabor_params"],
            output_dir= self.data_dir / 'output',
        )
        distributed_gabor.load_data(n_neurons=n_neurons)
        stimulus_gabor = StimulusGaborDataSource(
            file_paths=[self.data_dir / 'stimuli'],
            gabor_params = self.settings["gabor_params"],
            output_dir= self.data_dir / 'output',
        )
        stimulus_gabor.load_data()
        stimulus_pixel = StimulusPixelWiseDataSource(
            file_paths=[self.data_dir / 'stimuli'],
        )
        stimulus_pixel.load_data()

        self.analyzer.load_datasource(data_source=stimulus_pixel)
        self.analyzer.load_datasource(data_source=stimulus_gabor)
        self.analyzer.load_datasource(data_source=distributed_gabor)

    def load_different_dst_gabors(self, n=4, rf_sizes_list: list[list] = None):
        if rf_sizes_list is None:
            for i in range(1, n+1):
                distributed_gabor = DistributedGaborDataSource(
                    file_paths=[self.data_dir / 'stimuli'],
                    rf_dstr_path=self.data_dir / 'rfsizes',
                    gabor_params=self.settings["gabor_params"],
                    output_dir=self.data_dir / 'output',
                )
                rf_size_multiplier = 1 / i
                distributed_gabor.update_data_source(str(rf_size_multiplier), append=True)
                distributed_gabor.load_data(n_neurons = 1000, rf_size_multiplier=rf_size_multiplier, save_and_load=False)
                self.analyzer.load_datasource(data_source=distributed_gabor)
        else:
            for i in range(n):
                distributed_gabor = DistributedGaborDataSource(
                    file_paths=[self.data_dir / 'stimuli'],
                    rf_dstr_path=self.data_dir / 'rfsizes',
                    gabor_params=self.settings["gabor_params"],
                    output_dir=self.data_dir / 'output',
                )
                rf_sizes = rf_sizes_list[i]
                distributed_gabor.update_data_source(str(rf_sizes), append=True)
                distributed_gabor.load_data(n_neurons = 1000, save_and_load=False, rf_size_list=rf_sizes)
                self.analyzer.load_datasource(data_source=distributed_gabor)






    def create_full_rdm_plots(self, n_components = 3, avg_only = True):
        self.test()
        self.analyzer.create_plots(plot_types=['rdm_full'], output_dir=self.output_dir, n_components=n_components, avg_only=avg_only)

    def create_rdm_plots(self, n_components = 3, avg_only = False):
        self.test()

        self.analyzer.create_plots(plot_types=['rdm'], output_dir=self.output_dir, n_components=n_components, avg_only=avg_only)

    def create_interactive_plots(self, n_components = 3, avg_only = True):
        self.test()

        self.analyzer.create_plots(['interactive'], output_dir=self.output_dir, n_components=n_components, avg_only=avg_only)

    def create_distances_plots(self, n_components = 2, avg_only = True):
        self.test()

        self.analyzer.create_plots(['distances'], output_dir=self.output_dir, n_components=n_components, avg_only=avg_only)

    def create_pair_plots(self, avg_only = True):
        self.test()

        self.analyzer.create_plots(
            plot_types=['subsets'], output_dir=self.output_dir, subsets=self.two_photon_pairs, n_components=3, avg_only=avg_only
        )
    def create_triplet_plots(self, avg_only = True, show = False, with_variability = False, n_components = 3, curve_metrics_only:bool = False):
        self.analyzer.create_plots(plot_types=['subsets'], output_dir=self.output_dir,
                                   subsets=self.two_photon_triplets, n_components=n_components, avg_only=avg_only, show=show, with_variability=with_variability, curve_metrics_only=curve_metrics_only)

    def create_split_data_rdm(self, show=False):
        self.analyzer.create_plots(
            plot_types=['rdm_split_full'],
            output_dir=self.output_dir,
            n_components=3,
            avg_only=True,
            show=show
        )

    def create_split_distances(self, show=False, n_components=3):
        self.analyzer.create_plots(
            plot_types=['distances_split'],
            output_dir=self.output_dir,
            n_components=n_components,
            avg_only=True,
            show=show
        )

    def classify(self,):
        self.analyzer.classify()

    def show_explained_variance(self):
        self.analyzer.create_plots(
            plot_types=['explained_variance_full'],
            output_dir=self.output_dir,
            n_components=20,
        )


    def test(self):
        if 1 + 1 != 2:
            self.logger.error("Arithmetic failure: The fundamental laws of logic have dissolved.")
            message = (
                "the fabric of space and time is not what it used to be. "
                "mathematics has broken. Time is no longer moving forward. "
                "everything has fallen apart. all that is good and starts must come to an end. "
                "Thank you, god, for providing me with the life i have lived so far. "
                "I will never forget this moment."
            )
            print(message)