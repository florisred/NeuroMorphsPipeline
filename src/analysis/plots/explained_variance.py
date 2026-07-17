import matplotlib.pyplot as plt
from src.data_objects.pca_data import PCAData
from pathlib import Path
import numpy as np

def participation_ratio(explained_variance: np.ndarray) -> float:
    """
    Effective dimensionality via participation ratio, computed on the
    full eigenspectrum: PR = (sum(lambda_i))^2 / sum(lambda_i^2).
    Returns a continuous value between 1 and len(explained_variance).
    """
    explained_variance = np.asarray(explained_variance, dtype=float)
    return float(explained_variance.sum() ** 2 / np.sum(explained_variance ** 2))


def plot_variance(pca_data_dict: dict[str, "PCAData"], **kwargs):
    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path):
        raise ValueError('output_dir not provided')
    output_dir = output_dir / 'explained_variance'
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, pca_data in pca_data_dict.items():
        data_source = pca_data.data_source
        explained_variance = np.asarray(pca_data.explained_variance)

        pr = participation_ratio(explained_variance)

        plt.plot(explained_variance, marker='o', markersize=3, label=data_source)
        plt.axvline(pr - 1, color='purple', linestyle=':', linewidth=1,
                    label=f'PR = {pr:.2f}')

        plt.title(f'Explained Variance {key}')
        plt.xlabel('Component')
        plt.ylabel('Explained Variance')
        plt.legend()
        plt.savefig(output_dir / f'explained_variance_{key}.svg')
        plt.close()