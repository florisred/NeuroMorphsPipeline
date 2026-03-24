from pathlib import Path
from warnings import warn
from scripts.generate_config import generate_config
import json
from data_objects.two_photon_data_source import TwoPhotonDataSource
from analysis.analyzer import Analyzer
from data_objects.stimulus_data_source import StimulusGaborDataSource, StimulusPixelWiseDataSource

class Pipeline:

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
        self.session_dirs = [dire for dire in self.data_dir.iterdir() if dire.name not in ["output", "stimuli"] and not dire.name.startswith(".")]
        self.output_dir = self.data_dir / "output" / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = Analyzer()



    def load_two_photon(self, split:bool = False):
        two_photon = TwoPhotonDataSource(
            file_paths=self.session_dirs,
            data_location=self.settings["PSEUDOPOP_DATA"]
        )
        two_photon.load_data(split=split)
        self.two_photon_triplets = two_photon.find_stimulus_cycles(n=3)
        two_photon_pairs = two_photon.metadata.get_pair_keys(unique=True, dropna=True)
        self.two_photon_pairs = [[pair] for pair in two_photon_pairs]

        self.analyzer.load_datasource(data_source=two_photon)

    def load_stimuli(self):

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

    def create_plots(self):
        self.analyzer.create_plots([ 'interactive', 'distances'], output_dir=self.output_dir, n_components=6)
        self.analyzer.create_plots(
            plot_types=['rdm'], output_dir=self.output_dir, subsets=self.two_photon_pairs, n_components=2, avg_only = True
        )
        self.analyzer.create_plots(plot_types=['subsets', 'rdm'], output_dir=self.output_dir, subsets=self.two_photon_triplets, n_components=2, avg_only=True)

