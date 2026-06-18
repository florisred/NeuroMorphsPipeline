from src.pipeline import Pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    """
    Possible options, and what they do:
    
    First off, always load the pipeline: nm=Pipeline()
    
    Then, load DataSources within the pipeline, for further analysis.
    Possibilities:
     - nm.load_two_photon() -> loads the 2p data
        Possible arguments:
         - split: enables the splitting of trails into train and test sets
         - train_percent: if split is enabled, the percentage of the data used for the train dataset (0-1)
         
    - nm.load_stimuli() -> loads the PixelWiseDataSource, GaborDataSource, and DistributedGaborDataSource
        
        The PixelwiseDataSource just loads the pixel values of all the images, and performs PCA on that. 
            It represents the most direct way of processing the data
            
        The GaborDataSource works by processing the raw pixel values through a gabor filter bank.
            The parameters of this gabor filter bank can be adjusted in config.json. It creates a separate image for each
            permutation of the provided parameters, and then creates an 8x8 pixel value of these processed images.
            
        The DistributedGaborDataSource seeks to emulate the V1 area most closely. It is provided a n_neurons.
        Each of these neurons is given one of the parameters from config.json, and a receptive field.
        This receptive field sizes are provided in a file, also from the 2p acquisition process. It basically is a very
        long list of all the different sizes of the receptive fields that are mapped for all neurons found. 
        It then picks a random area of the image to create the receptive field based on a random size from the distribution.
        Each neuron is then processed independently, and is given an activation value based on the image provided.
        
        Possible arguments:
        n_neurons: The number of neurons in the DistributedGaborDataSource, default = 500
        
    
    - nm.load_different_dst_gabors:
        This function is handy if you want to load many different distributedGaborDataSources with different receptive
        field sizes, to see the effect of the size, distribution, etc, of receptive fields sizes.
        It loads n different DistributedGaborDataSources. By default, it will use the base distribution, and divide
        those values by i (in range n), to create smaller receptive field sizes.
        You can also pass it a list of lists of receptive field sizes, like so:
            nm.load_different_dst_gabors(n=5, rf_sizes_list=[[5,10], [10,20], [20,30], [30,40], [40,50]])
        This will create five distributed gabor datasources, one with receptive field sizes of 5 and 10, one with 
        10 and 20, etc. etc. These lists can be as long or short as you want.
        
        POSSIBLE arguments:
         - n: number of different DistributedGaborDataSources
         - rf_sizes_list: a list of lists of receptive field sizes
         
    Then, you can perform different analysis with this data:
    
    Possibilities:
        
        nm.classify():
            Trains a classifier based on the anchors (full morphs), then classifies the (partial) morphs accordingly.
            Shows a graph of all the different datasources, and how often the partial morphs are classified to the
            closest anchor
            
        nm.create_triplet_plots():
        
            A triplet is a transition for texture A->B, B->C, C->A. 
            When put into a PCA space, this creates a nice triangle. This can be done with all different DataSources.
            However, depending on the type of datasource, the morphs between the anchors will not follow the optimal
            straight line. Instead, there seems to be a curvature present.
            
            nm.create_triplet_plots() first finds all possible triplets in the 2p data, then plots these for all the 
            datasources loaded, 
            
    
    """

    nm = Pipeline()
    # nm.load_ori_two_photon()
    # nm.load_ori_pixel()
    # nm.load_ori_dist_gabor()
    # nm.load_ori_gabor()
    # nm.pca_ori_data()

    nm.load_two_photon(split=False)
    #
    # #nm.load_different_dst_gabors(n=5)#, rf_sizes_list=[[5,10], [10,20], [20,30], [30,40], [40,50]])
    #
    nm.load_stimuli(n_neurons=5612)

    #nm.create_interactive_plots()
    #nm.create_3d_plots()

    #nm.show_explained_variance()
    #
    # nm.classify()
    #
    #nm.create_triplet_plots(show=True, with_variability=True, curve_metrics_only = False)
    #
    nm.create_rdm_plots(n_components=6)
    #
    # nm.create_distances_plots()
    #
    #nm.create_interactive_plots()
    #
    #nm.create_anchor_rmd_plots(n_components=6)
    #nm.create_full_rdm_plots(n_components=6)


# ToDO: Hoe ver het allemaal van het midden afzit