from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from utils.utils import find_max_seperation_dataframe


def ori_ring_plot(
        ori_pca_data: pd.DataFrame,
        title: str,
        output_dir: Path
):

    components_to_use = find_max_seperation_dataframe(pca_dataframe=ori_pca_data, num_comps=2)
    orientation_data = ori_pca_data.index.astype(int).to_numpy()

    plt.figure()
    scatter = plt.scatter(
        ori_pca_data.iloc[:, components_to_use[0]],#components_to_use[0]],
        ori_pca_data.iloc[:, components_to_use[1]],
        c=orientation_data,
        cmap='viridis'
    )

    # 3. Add a colorbar to show the mapping from colors to orientation values
    plt.colorbar(scatter, label='Orientation Degree')
    plt.title(title)
    plt.savefig(output_dir / f'{title}_ori_ring.svg')
    test =1

def ori_explained_variance_plot(
        explained_variance: list,
        title: str,
        output_dir: Path
):
    plt.figure()
    plt.plot(explained_variance)
    plt.title(title)
    plt.savefig(output_dir / f'{title}_explained_variance.svg')