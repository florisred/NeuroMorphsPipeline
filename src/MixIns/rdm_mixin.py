import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import pandas as pd


class RDMMixIn:
    @classmethod
    def rdm_analysis(
            cls,
            pca_data_list: list,
            output_dir: Path,
            n_components: int = 2,
            dist_metric: str = 'euclidean'
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = output_dir / 'rdm'
        rdms = {}
        names = [d.name for d in pca_data_list]
        morph_names = pca_data_list[0].metadata.get_morph_names()

        # 1. Calculate RDMs
        for pca_data in pca_data_list:
            data = pca_data.get_data_components(n_components=n_components)
            rdms[pca_data.name] = pdist(data.values, metric=dist_metric)

        # 2. Representational Stability Matrix
        stability_matrix = np.zeros((len(names), len(names)))
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                stability_matrix[i, j], _ = spearmanr(rdms[n1], rdms[n2])

        cls._plot_stability(stability_matrix, names, output_dir)

        # 3. Individual RDMs
        for name, dist_vec in rdms.items():
            cls._plot_rdm(squareform(dist_vec), morph_names, name, output_dir)

    @staticmethod
    def _plot_stability(matrix, labels, output_dir):
        # We increase width to accommodate long session names
        plt.figure(figsize=(12, 10))

        # Using viridis (default behavior: higher correlation = brighter/yellower)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap='viridis',
                    xticklabels=labels, yticklabels=labels, square=True)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title("Representational Stability (Spearman ρ)", pad=20, fontsize=15)

        # tight_layout fixes the "cutoff" text on the left/bottom
        plt.tight_layout()
        plt.savefig(output_dir / "stability_matrix.png")
        plt.show()  # Forces the plot to show in the console/notebook

    @staticmethod
    def _plot_rdm(matrix, labels, name, output_dir):
        scale_factor = max(10, len(labels) * 0.3)
        plt.figure(figsize=(scale_factor, scale_factor))
        ax = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels,
                         cmap='rocket', square=True)
        if len(labels) > 20:
            n = len(labels) // 20  # Aim for ~20 labels total
            for i, label in enumerate(ax.xaxis.get_ticklabels()):
                if i % n != 0: label.set_visible(False)
            for i, label in enumerate(ax.yaxis.get_ticklabels()):
                if i % n != 0: label.set_visible(False)

        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.title(f"RDM: {name}", pad=20, fontsize=16)

        plt.tight_layout()
        plt.savefig(output_dir / f"rdm_{name}.png", bbox_inches='tight')
        plt.show()  # Forces the plot to show in the loop