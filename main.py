from src.pipeline import Pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":

    nm = Pipeline()

    nm.load_two_photon(split=False)

    nm.load_different_dst_gabors(n=5, rf_sizes_list=[[5,10], [10,20], [20,30], [30,40], [40,50]])

    #nm.load_stimuli(n_neurons=5612)

    #nm.classify()

    nm.create_triplet_plots(show=True, with_variability=True, curve_metrics_only = True)

    #nm.create_rdm_plots(n_components=3)

    #nm.create_distances_plots()

    #nm.create_interactive_plots()

    #nm.create_full_rdm_plots(n_components=3)

    ## ToDO:

    # maak op de x-as de RF-size,  y-as de curve metric op plaats 4
    # maak een deviation van de 2-p metric
    
