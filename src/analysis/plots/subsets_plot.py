from typing import Union
from data_objects.pca_data import PCAData
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from utils.utils import find_max_separation


def create_subset_plots(pca_data_dict: dict[str, PCAData], with_variability=False, **kwargs):
    components = find_max_separation(pca_data_dict=pca_data_dict, num_comps=2)

    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path):
        raise ValueError('output_dir not provided or not a Path object')

    output_dir = output_dir / 'subsets'
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, pca_data in pca_data_dict.items():
        if 'subset' not in key:
            continue

        pc_x, pc_y = components[key][0], components[key][1]

        data = pca_data.pca_data
        metadata = pca_data.metadata
        numeric_index = pca_data.get_numeric_index()

        plt.figure(figsize=(12.5, 7.5))
        ax = plt.gca()  # Get current axes to add the ellipse patches later

        plot_coords = data.iloc[:, [pc_x, pc_y]].values
        loop_data = np.vstack([plot_coords, plot_coords[0]])

        # Plot morph path
        plt.plot(loop_data[:, 0], loop_data[:, 1],
                 color='gray', alpha=0.4, linestyle='--', zorder=1, label='Morph Path')

        # Add variability (trial data & covariance clouds) if requested
        if with_variability and hasattr(pca_data, 'trial_data') and pca_data.trial_data is not None:
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=np.min(numeric_index), vmax=np.max(numeric_index))

            for i, name in enumerate(metadata.morph_names):
                try:
                    morph_trials = pca_data.trial_data.loc[name]

                    if isinstance(morph_trials, pd.Series):
                        trial_coords = morph_trials.iloc[[pc_x, pc_y]].values.reshape(1, -1)
                    else:
                        trial_coords = morph_trials.iloc[:, [pc_x, pc_y]].values

                    color = cmap(norm(numeric_index[i]))

                    # Plot the individual trial points
                    plt.scatter(trial_coords[:, 0], trial_coords[:, 1],
                                color=color, s=20, alpha=0.3, zorder=1, edgecolors='none')

                    # Calculate and plot the "Cloud" (Confidence Ellipse)
                    # We need at least 2 points to calculate covariance
                    if len(trial_coords) >= 2:
                        cov = np.cov(trial_coords[:, 0], trial_coords[:, 1])

                        # Ensure we have a valid 2D covariance matrix
                        if cov.ndim == 2:
                            # Calculate eigenvalues and eigenvectors to find the direction of variance
                            eigenvalues, eigenvectors = np.linalg.eigh(cov)
                            order = eigenvalues.argsort()[::-1]
                            eigenvalues = eigenvalues[order]
                            eigenvectors = eigenvectors[:, order]

                            # Calculate the angle for the ellipse
                            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

                            # Calculate width and height for 1.5 standard deviations
                            n_std = 1
                            width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 0))

                            # Create and add the ellipse patch
                            ell = Ellipse(xy=(np.mean(trial_coords[:, 0]), np.mean(trial_coords[:, 1])),
                                          width=width, height=height, angle=angle,
                                          facecolor=color, alpha=0.15, edgecolor=color,
                                          linewidth=1.5, zorder=1)
                            ax.add_patch(ell)

                except KeyError:
                    pass

        # Plot average points
        plt.scatter(plot_coords[:, 0], plot_coords[:, 1],
                    c=numeric_index, cmap='viridis', s=60, alpha=0.8,
                    edgecolors='white', zorder=2)

        is_anchor = metadata.anchor_mask
        for i in np.where(is_anchor)[0]:
            name = metadata.morph_names.iloc[i]
            plt.text(plot_coords[i, 0], plot_coords[i, 1] + 0.5, name, fontsize=10,
                     fontweight='bold', ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))

        cbar = plt.colorbar()
        cbar.set_ticks(numeric_index)
        cbar.set_ticklabels(metadata.morph_names)
        plt.xlabel(f'PC{pc_x + 1}')
        plt.ylabel(f'PC{pc_y + 1}')
        plt.title(f'{pca_data.name} 2D (PC{pc_x + 1} vs PC{pc_y + 1})')

        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(output_dir / f'{pca_data.name}.png')
        plt.close()