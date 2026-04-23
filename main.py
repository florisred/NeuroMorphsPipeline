from src.pipeline import Pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":

    nm = Pipeline()
    nm.load_two_photon(split=False)
    #nm.load_two_photon(split=True, train_percent = 0.3)
    # nm.create_split_data_rdm(show=True)
    # nm.create_split_distances(show=True,n_components=20)
    nm.load_stimuli(n_neurons=5612)

    #nm.show_explained_variance()

    #nm.classify()
    #
    #
    #nm.create_pair_plots()
    #
    nm.create_triplet_plots(show=True, with_variability=True)
    #
    #
    # #
    nm.create_rdm_plots(n_components=3)
    # #
    nm.create_distances_plots()
    # #
    nm.create_interactive_plots()
    #
    nm.create_full_rdm_plots(n_components=3)


## ToDO: Implement kfolds

#          8.4
# //            7.94,
# //            8.4,
# //            11.84,
# //            16.81,
# //            23.78