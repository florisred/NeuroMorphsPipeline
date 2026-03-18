from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from data_objects.pca_data import PCAData


def create_subset_plots(pca_data_dict: dict[str, PCAData], **kwargs):
    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path): raise ValueError('output_dir not provided')
    output_dir = output_dir / 'subsets'
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in pca_data_dict.keys():
        if 'subset' not in key: continue
        pca_data = pca_data_dict[key]
        output_dir.mkdir(parents=True, exist_ok=True)
        data = pca_data.pca_data
        metadata = pca_data.metadata
        numeric_index = pca_data.get_numeric_index()

        plt.figure(figsize=(12.5, 7.5))
        loop_data = np.vstack([data, data.iloc[0]])
        plt.plot(loop_data[:, 0], loop_data[:, 1],
                 color='gray', alpha=0.4, linestyle='--', zorder=1, label='Morph Path')
        sc = plt.scatter(data.iloc[:, 0], data.iloc[:, 1],
                         c=numeric_index, cmap='viridis', s=60, alpha=0.8,
                         edgecolors='white', zorder=2)
        is_anchor = metadata.get_anchor_mask()
        for i in np.where(is_anchor)[0]:
            name = metadata.get_morph_names().iloc[i]
            plt.text(data.iloc[i, 0], data.iloc[i, 1] + 0.5, name, fontsize=10,
                     fontweight='bold', ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))
        cbar = plt.colorbar(sc)
        cbar.set_ticks(numeric_index)
        cbar.set_ticklabels(metadata.get_morph_names())
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'{pca_data.name} 2D')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(output_dir / f'{pca_data.name}.png')
        plt.show()