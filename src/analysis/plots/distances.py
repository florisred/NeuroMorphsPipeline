from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from data_objects.pca_data import PCAData

def calculate_distances(pca_data_dict: dict[str, PCAData], **kwargs):
    # start and validate
    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path): raise ValueError('output_dir not provided')
    output_dir = output_dir / 'distances'
    output_dir.mkdir(parents=True, exist_ok=True)
    x = None

    all_distances = defaultdict(list)
    all_distances_cum = defaultdict(list)

    for k, pca_data in pca_data_dict.items():
        if 'subset' not in k: continue
        data_source = pca_data.data_source
        data = pca_data.pca_data
        distances_sqr = squareform(pdist(X=data,metric='euclidean'))

        cumsum_differences = distances_sqr[0]
        distances = np.diag(distances_sqr, k=1)

        scaled_distances_cum = [dis / cumsum_differences[-1] for dis in cumsum_differences]
        scaled_distances = [dis / distances[-1] for dis in distances]
        all_distances[data_source].append(scaled_distances)
        all_distances_cum[data_source].append(scaled_distances_cum)
        if x is None:
            x = pca_data.metadata.morph_steps
            x.iloc[0] = 0
            x.iloc[-1] = 1


    for data_source in all_distances.keys():
        mean_distance = np.mean(all_distances[data_source], axis=0)
        sem_distance = np.std(all_distances[data_source], axis=0)
        mean_distance_cum = np.mean(all_distances_cum[data_source], axis=0)
        sem_distance_cum = np.std(all_distances_cum[data_source], axis=0)


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
        title = f"Mean Distances scaled {data_source}"
        plt.legend()
        plt.title(title)
        plt.savefig(output_dir / f'{title}.png')
        plt.show()

        # Plot 2: Cumulative Mean Distances
        plt.figure(figsize=(10, 7.5))
        plt.plot(x, mean_distance_cum, label='Mean Cumulative', color='green')
        # Adding the SEM/STD ribbon
        plt.fill_between(
            x,
            mean_distance_cum - sem_distance_cum,
            mean_distance_cum + sem_distance_cum,
            color='green', alpha=0.2
        )
        title = f"Mean Distance (cumulative) {data_source}"
        plt.title(title)
        plt.savefig(output_dir / f'{title}.png')
        plt.show()