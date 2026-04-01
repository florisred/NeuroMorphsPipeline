from typing import Union
from data_objects.pca_data import PCAData
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import find_max_separation

def create_subset_plots(pca_data_dict: dict[str, PCAData], components: Union[list[int], str] = 'auto',  **kwargs):


    if components == 'auto' or True:
        components = find_max_separation(pca_data_dict=pca_data_dict, num_comps=2)

    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path):
        raise ValueError('output_dir not provided or not a Path object')

    output_dir = output_dir / 'subsets'
    output_dir.mkdir(parents=True, exist_ok=True)

    pc_x, pc_y = components[0], components[1]

    for key, pca_data in pca_data_dict.items():
        if 'subset' not in key:
            continue

        data = pca_data.pca_data
        metadata = pca_data.metadata
        numeric_index = pca_data.get_numeric_index()

        plt.figure(figsize=(12.5, 7.5))

        # 1. Handle looping the data using dynamic components
        # We extract just the 2 selected columns to make vstacking clean
        plot_coords = data.iloc[:, [pc_x, pc_y]].values
        loop_data = np.vstack([plot_coords, plot_coords[0]])

        plt.plot(loop_data[:, 0], loop_data[:, 1],
                 color='gray', alpha=0.4, linestyle='--', zorder=1, label='Morph Path')

        # 2. Scatter plot using dynamic components
        plt.scatter(plot_coords[:, 0], plot_coords[:, 1],
                    c=numeric_index, cmap='viridis', s=60, alpha=0.8,
                    edgecolors='white', zorder=2)

        # 3. Dynamic text placement
        is_anchor = metadata.anchor_mask
        for i in np.where(is_anchor)[0]:
            name = metadata.morph_names.iloc[i]
            # Coordinates extracted dynamically based on requested components
            plt.text(plot_coords[i, 0], plot_coords[i, 1] + 0.5, name, fontsize=10,
                     fontweight='bold', ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))

        # 4. Colorbar and Labels
        cbar = plt.colorbar()
        cbar.set_ticks(numeric_index)
        cbar.set_ticklabels(metadata.morph_names)

        # Dynamic axis labels (PC1, PC2 or PC0, PC1 depending on your naming preference)
        plt.xlabel(f'PC{pc_x + 1}')
        plt.ylabel(f'PC{pc_y + 1}')
        plt.title(f'{pca_data.name} 2D (PC{pc_x + 1} vs PC{pc_y + 1})')

        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(output_dir / f'{pca_data.name}.png')
        plt.close()