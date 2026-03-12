import matplotlib.pyplot as plt
import numpy as np

from analysis.PCAData import PCAData


class Plot2DMixIn:

    @staticmethod
    def create_2d_plots(pca_data: PCAData):
        data = pca_data.pca_data
        numeric_index = pca_data.

        plt.figure(figsize=(12.5, 7.5))
        loop_data = np.vstack([data, data[0]])
        plt.plot(loop_data[:, 0], loop_data[:, 1],
                 color='gray', alpha=0.4, linestyle='--', zorder=1, label='Morph Path')
        sc = plt.scatter(data[:, 0], data[:, 1],
                         c=numeric_index, cmap='viridis', s=60, alpha=0.8,
                         edgecolors='white', zorder=2)
        is_anchor = labels['stim_type'] == 'anchor'
        for i in np.where(is_anchor)[0]:
            name = labels['morph_name'].iloc[i]
            plt.text(data[i, 0], data[i, 1] + 0.5, name, fontsize=10,
                     fontweight='bold', ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))
        cbar = plt.colorbar(sc)
        cbar.set_ticks(numeric_index)
        cbar.set_ticklabels(labels['morph_name'])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'{self.plot_name} 2D {self.pca_type} (Closed Loop)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()