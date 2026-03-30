from src.pipeline import Pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":

    nm = Pipeline()
    #nm.load_two_photon(split=True)
    nm.load_two_photon(split=False)
    #nm.create_split_data_rdm(show=True)
    #nm.create_split_distances(show=True,n_components=20)

    #nm.load_stimuli()

    #nm.create_pair_plots()

    #nm.create_triplet_plots()

    #nm.create_full_rdm_plots()

    #nm.create_distances_plots()

    nm.create_interactive_plots()
