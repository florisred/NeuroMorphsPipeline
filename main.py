
from src.pipeline import Pipeline

if __name__ == "__main__":

    nm = Pipeline()

    nm.process_two_photon()
    nm.process_stimuli()
    nm.run_pca()
    nm.run_rdm()



#     nm.load_neuropixels_data()


    nm.create_plots()



