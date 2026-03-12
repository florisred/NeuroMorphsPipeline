
from src.pipeline import Pipeline
import os
from datetime import datetime

if __name__ == "__main__":

    nm = Pipeline()
    nm.test()


    nm.process_two_photon()
    nm.process_stimuli()
    nm.run_pca()

#     nm.load_neuropixels_data()


    nm.create_plots()


