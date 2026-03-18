from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from data_objects.pca_data import PCAData

def calculate_distances(pca_data_dict: dict[str, PCAData], **kwargs):
    # start and validate
    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path): raise ValueError('output_dir not provided')
    output_dir = output_dir / 'distances'
    output_dir.mkdir(parents=True, exist_ok=True)

    for k, pca_data in pca_data_dict.items():
        if 'full' not in k: continue
        data = pca_data.pca_data
        metadata = pca_data.metadata
        name_stems = metadata.get_pair_keys(unique=True, values=True, dropna=True)
        all_distances = []
        all_distances_cum = []
        pair_keys = metadata.get_pair_keys(unique=False, values=False, dropna=False)

        for stem in name_stems:
            mask = pair_keys == stem
            anchor_mask = metadata.get_morph_names().isin(stem.split('__'))
            mask = mask | anchor_mask
            pca_masked = data[mask]
            distances = [np.linalg.norm(pca_masked.iloc[i + 1] - pca_masked.iloc[i]) for i in range(len(pca_masked) - 1)]
            distances_with_0 = [0.0]
            for distance in distances: distances_with_0.append(distance)
            cumsum_differences = np.cumsum(distances_with_0)
            scaled_distances_cum = [dis / cumsum_differences[-1] for dis in cumsum_differences]
            scaled_distances = [dis / cumsum_differences[-1] for dis in distances]
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
        title = f"Mean Distances scaled {pca_data.name}"
        plt.ylim(0, 1)
        plt.legend()
        plt.title(title)
        plt.savefig(output_dir / f'{title}.png')
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
        title = f"Mean Distance (cumulative) {pca_data.name}"
        plt.title(title)
        plt.savefig(output_dir / f'{title}.png')
        plt.show()