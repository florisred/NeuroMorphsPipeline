from pathlib import Path
from os.path import join, isdir
from warnings import warn
from src.generate_config import generate_config
from src.stimuli import Stimuli
import json

class Pipeline:

    def __init__(self):
        # initiate stimuli object for storing the stimuli

        self.stimuli = Stimuli()
        self.neuropixels_data = NeuropixelsData()

        # set preliminary vars
        project_root = Path(__file__).parent
        config_dir = project_root / "config"
        settings_file = config_dir / "settings.json"


        # check if there is a config file, if not let user generate it
        if not settings_file.is_file():
            warn("Configuration file not found.")
            if (input("Make config file from defaults? (y/n)").lower() == "y"):
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

        self.stimuli.set_data_dir(self.settings["DATA_FOLDER"])




    def load_images(self):
        """

        :return:
        """
        self.stimuli.load_images()

    def process_images(self):
        self.stimuli.process_images(
            n_components_pixel = self.settings["num_components_pixel_space"],
            n_components_gabor = self.settings["num_components_gabor_space"],
            gabor_params = self.settings["gabor_params"]
        )

    def create_plots(self):

        if len(self.stimuli.pca_dict.keys()) > 0:
            for key, pca_list in self.stimuli.pca_dict.items():
                print(f"Creating plot for {key}...")
                pca_data = pca_set[0]
                explained
                pass




    def load_neuropixels_data(self):
        self.neuropixels_data.load_data()
