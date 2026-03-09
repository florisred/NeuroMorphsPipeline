import numpy as np
import pandas as pd

from helper.PlotCreatorMixInHelper import PlotCreatorMixInHelper


class PlotCreatorMixIn(PlotCreatorMixInHelper):

    def __init__(self, plot_settings):
        self.plot_name = plot_settings["plot_name"]
        self.do_2d_plots = plot_settings["do_2d_plots"]
        self.do_3d_plots = plot_settings["do_3d_plots"]
        self.do_distances = plot_settings["do_distances"]
        self.do_interactive_plots = plot_settings["do_interactive_plots"]
        self.pca = []
        self.labels = pd.DataFrame()
        self.name_mask = []
        self.pca_type = None
        self.output_dir = plot_settings["data_dir"]



    def create_plots(self, pca, processed_names, pca_type, name_mask=None):


        #pca = pca.to_numpy()
        self.pca = pca
        self.labels = processed_names #pd.DataFrame(processed_names)
        if name_mask is None:
            name_mask = np.zeros(len(processed_names), dtype=bool)
        self.name_mask = name_mask
        self.pca_type = pca_type





        print("Creating plots...")
        is_full = 'full' in self.pca_type
        if self.do_distances and is_full:
            self.calculate_distances()
        if self.do_2d_plots and not is_full:
            self._create_2d_plots()
        if self.do_3d_plots and is_full:
            self._create_3d_plots()
        if self.do_interactive_plots and is_full:
            self._create_interactive_plot()