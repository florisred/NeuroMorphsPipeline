from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def ori_ring_plot(
        ori_pca_data: pd.DataFrame,
        title: str,
        output_dir: Path
):

    orientation_data = ori_pca_data.index.astype(int).to_numpy()

    plt.figure()
    scatter = plt.scatter(
        ori_pca_data.iloc[:, 0],
        ori_pca_data.iloc[:, 1],
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