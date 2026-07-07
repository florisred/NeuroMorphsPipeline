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
        data_folder= "/Users/floris/Documents/neuro_texmorphs/data", # texture
        #data_folder = "/Users/floris/Documents/OrientationData/Ori_NeuralData/Ori_Data" # orientation data
    )
    nm.load_data_sources(
        sources=[
            '2p',
            'GaborFilterBank',
            'GaborNet',
            'RetinoDivNormGaborNet',
            'PixelWise',
            # 'ori_2p',
            # 'ori_GaborFilterBank',
            # 'ori_RetinodivnormGaborNet',
            # 'ori_GaborNet',
            # 'ori_PixelWise',
            # 'ori_pca'
        ],
        h5_locations_textures = {
            'data':'X',
            'labels': ['y/pair_key', 'y/step_index', 'y/src_cat', 'y/dst_cat']
        },
        h5_locations_ori = {
            'data':'X',
            'labels': ['y/orientation_deg']
        },
        gabornet_params = {
            'n_neurons': 5000,
            'n_trials': 7,
            'recalculate_gabornet': True,
            'fano_factor': 1.2,
            'sensor_noise_std': 0.1,
            "orientation_dict": {
                "0": 0.18,
                "15": 0.104,
                "30": 0.066,
                "45": 0.043,
                "60": 0.053,
                "75": 0.066,
                "90": 0.097,
                "105": 0.106,
                "120": 0.087,
                "135": 0.087,
                "150": 0.058,
                "165": 0.053
            },
            "gamma": 0.5,
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
            #'classification',
            #'explained_variance_full',
            #'explained_variance_subsets'
        ],
        plot_config = {
            #'show': False, # decides if the plots are shown in the IDE or not, they are always saved
            'subsets_n' : 3, # decides the number of anchors in the subsets plot, use 3 for the triplets
            'subsets_with_variability' : False, # if there are multiple trials, plot all of them next to the average
            'n_components' : 3,
            'PCA_on_anchors': False # If true, the PCA space will be created by only looking at the anchors (so 3 for triplets, or 8 for full data)
        }
    )



