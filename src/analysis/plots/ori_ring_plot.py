from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from utils.utils import find_max_seperation_dataframe

# --- Set Global Font Sizes ---
# This increases the font size for all text elements globally
plt.rcParams.update({
    'font.size': 14,  # Controls default text sizes
    'axes.titlesize': 16,  # Size of the plot title
    'axes.labelsize': 14,  # Size of the x and y labels
    'xtick.labelsize': 12,  # Size of the x-axis tick labels
    'ytick.labelsize': 12,  # Size of the y-axis tick labels
    'figure.titlesize': 18  # Size of the figure title (if used)
})


def ori_ring_plot(
        ori_pca_data: pd.DataFrame,
        title: str,
        output_dir: Path
):
    components_to_use = find_max_seperation_dataframe(pca_dataframe=ori_pca_data, num_comps=2)
    orientation_data = ori_pca_data.index.astype(int).to_numpy()

    plt.figure()
    scatter = plt.scatter(
        ori_pca_data.iloc[:, components_to_use[0]],
        ori_pca_data.iloc[:, components_to_use[1]],
        c=orientation_data,
        cmap='twilight'
    )

    # The colorbar and title will automatically use the global sizes set above
    cbar = plt.colorbar(scatter)
    cbar.set_label('Orientation Degree', fontsize=14)
    cbar.ax.tick_params(labelsize=12)  # Increases colorbar tick size

    plt.title(title)
    plt.xlabel(f'PC {components_to_use[0]+1}')
    plt.ylabel(f'PC {components_to_use[1]+1}')

    plt.tight_layout()
    plt.savefig(output_dir / f'{title}_ori_ring.svg')


def ori_explained_variance_plot(
        explained_variance: list,
        title: str,
        output_dir: Path
):
    plt.figure()
    plt.plot(explained_variance)

    plt.title(title)
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance')

    plt.tight_layout()  # Keeps everything neatly contained
    plt.savefig(output_dir / f'{title}_explained_variance.svg')