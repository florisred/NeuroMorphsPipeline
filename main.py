from src.pipeline import Pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    """     
    Only load either ori data or texture data, not both!
    """

    nm = Pipeline(
        data_folder= "/home/protred/Documents/neuro_texmorphs/data/", # texture
        # data_folder = ""/home/protred/Documents/OrientationData/Ori_NeuralData/Ori_Data/" # orientation data
    )
    nm.load_data_sources(
        sources=[
            '2p',
            'GaborFilterBank',
            'GaborNet',
            'PixelWise',
            #'ori_2p',
            #'ori_GaborFilterBank',
            #'ori_GaborNet',
            #'ori_PixelWise',
        ],
        n_simulated_neurons_GaborNet = 5612,
        recalculate_gabornet = False,
        n_repeat_trials=7,
        h5_locations_textures = {
            'data':'X',
            'labels': ['y/pair_key', 'y/step_index', 'y/src_cat', 'y/dst_cat']
        },
        h5_locations_ori = {
            'data':'X',
            'labels': 'y/orientation_deg'
        }
    )

    nm.create_plots(
        plot_types = [
            'interactive',
            'subsets',
            '3d',
            'rdm_subsets',
            'rdm_full',
            'anchor_rdm',
            # 'classification',
            # 'explained_variance_full',
            # 'explained_variance_subsets'
        ],
        plot_config = {
            'show' : False, # decides if the plots are shown in the IDE or not, they are always saved
            'subsets_n' : 3, # decides the number of anchors in the subsets plot, use 3 for the triplets
            'subsets_with_variability' : True, # if there are multiple trials, plot all of them next to the average
            'n_components' : 10
        }
    )


    #nm.create_interactive_plots()
    #nm.create_3d_plots()

    #nm.show_explained_variance()
    #
    # nm.classify()
    #
    #nm.create_triplet_plots(show=True, with_variability=True)
    #
    #nm.create_rdm_plots(n_components='max')
    #
    # nm.create_distances_plots()
    #
    #nm.create_interactive_plots()
    #
    # nm.create_anchor_rmd_plots(n_components='max')
    # nm.create_full_rdm_plots(n_components='max')


# ToDO: Hoe ver het allemaal van het midden afzit