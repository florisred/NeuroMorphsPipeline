from pathlib import Path
from warnings import warn
from src.two_photon import TwoPhoton
from plotcreator import PlotCreator
from scripts.generate_config import generate_config
from src.stimuli import Stimuli
from src.neuropixels import NeuroPixelsData
import os
import json
from src.rdm import RepresentationalDissimilarityMatrix

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


        data_dir = Path(self.settings["DATA_FOLDER"])
        self.session_dirs = [dire for dire in data_dir.iterdir() if dire.name not in ["output", "stimuli"] and not dire.name.startswith(".")]

        self.two_photon = TwoPhoton()
        self.stimuli = Stimuli(
            data_dir=self.settings["DATA_FOLDER"]
        )

    def process_stimuli(self):

        test=1




    def run_rdm(self):
        rdm = RepresentationalDissimilarityMatrix(pca_dict=self.pca_dict)
        rdm.run_rdm()



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
        self.data_dict['two-photon'] = self.two_photon.load_2p_data(
            session_dirs=self.session_dirs,
            data_location=self.settings["PSEUDOPOP_DATA"]
        )
