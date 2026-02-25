
from src.pipeline import Pipeline
import os
from datetime import datetime

if __name__ == "__main__":

    nm = Pipeline()
    print("looking for process_me.flag...")
    for root, directory, files in os.walk(nm.settings["DATA_FOLDER"]):
        if 'process_me.flag' in files:
            print(f'Found process flag in {root}. \n Starting processing at {datetime.now().strftime("%H:%M:%S")}')

            # look for 2p_data folder
            if '2p_data' in directory:
                nm.process_two_photon()
                nm.create_plots()
                ## process 2p data


            if 'neuropixels_data' in directory:

                nm.load_neuropixels_data()
                nm.create_plots()


            # look for image stimuli data
            if 'stimuli' in directory:

                nm.load_images()

                nm.process_images()

                nm.create_plots()
