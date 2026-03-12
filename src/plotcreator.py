import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import seaborn as sns


class PlotCreator:
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







    def calculate_distances(self):
        name_stems = self.labels['pair_key'].dropna().values
        all_distances = []
        all_distances_cum = []
        for stem in np.unique(name_stems):
            mask = self.labels['pair_key'] == stem
            anchor_mask = self.labels['morph_name'].isin(stem.split('__'))
            mask = mask | anchor_mask
            pca_masked = self.pca[mask, :2]
            distances = [np.linalg.norm(pca_masked[i+1] - pca_masked[i]) for i in range(len(pca_masked)-1)]
            cumsum_differences = np.cumsum(distances)
            scaled_distances_cum = [dis / cumsum_differences[-1] for dis in cumsum_differences]
            scaled_distances = [dis / cumsum_differences[-1] for dis in distances]
            # plt.plot(scaled_distances_cum)
            # plt.title(f'{label} distances between points (cumulative) {self.pca_type}')
            # plt.show()
            all_distances.append(scaled_distances)
            all_distances_cum.append(scaled_distances_cum)

        mean_distance = np.mean(all_distances, axis=0)
        sem_distance = np.std(all_distances, axis=0)
        mean_distance_cum = np.mean(all_distances_cum, axis=0)
        sem_distance_cum = np.std(all_distances_cum, axis=0)

        # Plot 1: Mean Distances
        plt.figure(figsize=(10, 7.5))
        plt.plot(mean_distance, label='Mean Distance', color='blue')
        # Adding the SEM/STD ribbon
        plt.fill_between(
            range(len(mean_distance)),
            mean_distance - sem_distance,
            mean_distance + sem_distance,
            color='blue', alpha=0.2, label='± 1 STD'
        )
        plt.title(f"Mean Distances scaled {self.pca_type}")
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

        # Plot 2: Cumulative Mean Distances
        plt.figure(figsize=(10, 7.5))
        plt.plot(mean_distance_cum, label='Mean Cumulative', color='green')
        # Adding the SEM/STD ribbon
        plt.fill_between(
            range(len(mean_distance_cum)),
            mean_distance_cum - sem_distance_cum,
            mean_distance_cum + sem_distance_cum,
            color='green', alpha=0.2
        )
        plt.title(f"Mean Distance (cumulative) {self.pca_type}")
        plt.show()
        tes = 1

