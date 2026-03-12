from pathlib import Path
from warnings import warn
from src.three_photon import Twophoton
from plotcreator import PlotCreator
from scripts.generate_config import generate_config
from src.stimuli import Stimuli
from src.neuropixels import NeuroPixelsData
import os
import json
from src.pca import PCAPerformer
from data_loader.TwoPhotonDataSource import TwoPhotonDataSource
from analysis.Analyzer import Analyzer
from data_loader.StimulusDataSource import StimulusGaborDataSource, StimulusPixelWiseDataSource

class Pipeline:

    def __init__(self):
        # initiate stimuli object for storing the stimuli


        self.pca_dict = {}
        self.data_dict = {}

        # set preliminary vars
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


    def test(self):
        two_photon = TwoPhotonDataSource(
            file_paths=self.session_dirs,
            data_location=self.settings["PSEUDOPOP_DATA"]
        )
        two_photon.load_data()
        analyzer = Analyzer()
        analyzer.run_pca(data_source=two_photon, pca_type='full')
        analyzer.run_pca(data_source=two_photon, pca_type='triplets')
        test=1

        # transitions = two_photon.find_stimulus_cycles(n=3)[0]
        # transition_data, transition_labels = two_photon.filter_transitions(transitions)
        #
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
        analyzer.run_pca(data_source=stimulus_pixel, pca_type='full')
        analyzer.run_pca(data_source=stimulus_pixel, pca_type='triplets')
        analyzer.run_pca(data_source=stimulus_gabor, pca_type='full')
        analyzer.run_pca(data_source=stimulus_gabor, pca_type='triplets')
        test = 1




    def process_stimuli(self):
        stimuli = Stimuli(
            data_dir = self.settings["DATA_FOLDER"]
        )
        self.data_dict.update(stimuli.process_images(
            gabor_params = self.settings["gabor_params"]
        ))

    def run_pca(self):
        pca_performer = PCAPerformer(data_dict=self.data_dict)
        self.pca_dict = pca_performer.run_pca_analysis()
        test = 1



    def create_plots(self):
        output_folder = Path(self.settings["DATA_FOLDER"]) / 'output' / 'plots'
        os.makedirs(output_folder, exist_ok=True)
        if len(self.pca_dict.keys()) > 0:
            for key, pca in self.pca_dict.items():
                print(f"Creating plot for {key}...")
                plotcreator = PlotCreator(plot_settings = {
                    "plot_name" : f'{key}',
                    'do_2d_plots': True,
                    'do_3d_plots': True,
                    'do_interactive_plots': True,
                    'do_distances': True,
                    'data_dir': output_folder,
                })
                plotcreator.create_plots(pca[0], pca[2], key)




    def load_neuropixels_data(self):
        neuropixels_data = NeuroPixelsData()
        neuropixels_data.load_data(data_dir=self.settings["DATA_FOLDER"])
        _, self.pca_dict["neuropixels"], self.neuropixels_labels = neuropixels_data.process_neuropixels_data()


    def process_two_photon(self):
        two_photon = Twophoton()
        self.data_dict['two_photon'] = two_photon.load_2p_data(
            session_dirs=self.session_dirs,
            data_location=self.settings["PSEUDOPOP_DATA"]
        )
        test=1