import matplotlib.pyplot as plt
from src.data_objects.pca_data import PCAData
from pathlib import Path

def plot_variance(pca_data_dict: dict[str, PCAData], **kwargs):
    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path):
        raise ValueError('output_dir not provided')
    output_dir = output_dir / 'explained_variance'
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, pca_data in pca_data_dict.items():
        data_source = pca_data.data_source
        explained_variance = pca_data.explained_variance

        plt.plot(explained_variance, label=data_source)
        plt.title(f'Explained Variance {key}')
        plt.savefig(output_dir / f'explained_variance_{key}.svg')
        plt.close()
