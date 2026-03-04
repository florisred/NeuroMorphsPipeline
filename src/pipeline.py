from pathlib import Path
from warnings import warn
from src.two_photon import Twophoton
from plotcreator import PlotCreator
from scripts.generate_config import generate_config
from src.stimuli import Stimuli
from src.neuropixels import NeuroPixelsData
import os
import json

class Pipeline:

    def __init__(self):
        # initiate stimuli object for storing the stimuli

        self.stimuli = Stimuli()
        self.neuropixels_data = NeuroPixelsData()
        self.two_photon = Twophoton()
        self.pca_dict = {}
        self.neuropixels_labels = None

        # set preliminary vars
        project_root = Path(__file__).parent.parent
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



    def process_stimuli(self):
        self.stimuli.load_images()
        pca_dict = self.stimuli.process_images(
            n_components_pixel = self.settings["num_components_pixel_space"],
            n_components_gabor = self.settings["num_components_gabor_space"],
            gabor_params = self.settings["gabor_params"]
        )
        self.pca_dict["pixel"] = pca_dict["pixel"]
        self.pca_dict["gabor"] = pca_dict["gabor"]



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
        self.neuropixels_data.load_data(data_dir=self.settings["DATA_FOLDER"])
        _, self.pca_dict["neuropixels"], self.neuropixels_labels = self.neuropixels_data.process_neuropixels_data()


    def process_two_photon(self):
        self.two_photon.load_2p_data(
            data_dir=self.settings["DATA_FOLDER"],
            data_location=self.settings["PSEUDOPOP_DATA"],
            label_location=self.settings["PSEUDOPOP_LABELS"]
        )
        self.pca_dict.update(self.two_photon.partial_pca_full_morphs(choose_transitions=False))
        self.pca_dict["two_photon"] = self.two_photon.peform_pca()
