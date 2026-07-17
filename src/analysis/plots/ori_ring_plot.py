from utils.utils import find_max_seperation_dataframe
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

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




def analyze_periodic_spacing(
        ori_pca_data: pd.DataFrame,
        title: str,
        output_dir: Path,
        num_bootstraps: int = 1000,
        alpha: float = 0.05,
        random_state: int = 42,
        n_compoments:str|int = 'max'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quantifies and compares high-dimensional Euclidean distances between adjacent
    periodic orientation values to test for manifold uniformity. Safe for
    Pandas 2.0+/3.0+ and optimized for publication-quality visual legibility.

    Parameters:
        ori_pca_data: DataFrame with PCA coordinates. Index must contain numerical
                      orientation values.
        title: Run identifier for saving plots and data.
        output_dir: Path to save the output files.
        num_bootstraps: Number of bootstrap iterations for standard error and hypothesis testing.
        alpha: False Discovery Rate (FDR) significance threshold.
        random_state: Seed for reproducibility.
        n_compoments: use 'max' (default) for no max components, otherwise an int for the number of components to explore

    Returns:
        step_summary_df: Summary of observed distances and 95% Confidence Intervals.
        step_comparison_df: Pairwise matrix of FDR-adjusted p-values comparing transition lengths.
    """
    if n_compoments != 'max':
        ori_pca_data = ori_pca_data.iloc[:, :n_compoments]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(random_state)

    # ----------------------------------------------------
    # 1. Parse and Sort Periodic Orientation Groups
    # ----------------------------------------------------
    # Cast index to float to ensure correct numerical sorting of orientation angles
    unique_orients = np.sort(np.unique(ori_pca_data.index.to_numpy().astype(float)))
    n_groups = len(unique_orients)

    if n_groups < 3:
        raise ValueError("Need at least 3 unique orientation values to evaluate periodic spacing.")

    # Group data into dictionary of raw numpy arrays for optimized bootstrapping
    group_data = {
        g: ori_pca_data.loc[ori_pca_data.index.astype(float) == g].to_numpy()
        for g in unique_orients
    }

    # Generate human-readable transition labels (e.g., "0.0°→15.0°")
    step_labels = []
    for i in range(n_groups):
        next_i = (i + 1) % n_groups
        step_labels.append(f"{unique_orients[i]}°→{unique_orients[next_i]}°")

    # ----------------------------------------------------
    # 2. Observed Step-wise Distances
    # ----------------------------------------------------
    obs_centroids = np.array([group_data[g].mean(axis=0) for g in unique_orients])
    obs_distances = np.zeros(n_groups)
    for i in range(n_groups):
        next_i = (i + 1) % n_groups
        obs_distances[i] = np.linalg.norm(obs_centroids[i] - obs_centroids[next_i])

    # ----------------------------------------------------
    # 3. Bootstrapping Spacing Distribution
    # ----------------------------------------------------
    print(f"Bootstrapping {num_bootstraps} iterations to analyze periodic spacing...")
    boot_distances = np.zeros((num_bootstraps, n_groups))

    for b in range(num_bootstraps):
        boot_centroids = []
        for g in unique_orients:
            arr = group_data[g]
            n_samples = len(arr)
            # Resample with replacement
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_centroids.append(arr[boot_idx].mean(axis=0))

        boot_centroids = np.array(boot_centroids)
        for i in range(n_groups):
            next_i = (i + 1) % n_groups
            boot_distances[b, i] = np.linalg.norm(boot_centroids[i] - boot_centroids[next_i])

    # Compute descriptive statistics from bootstrap runs
    ci_lower = np.percentile(boot_distances, 2.5, axis=0)
    ci_upper = np.percentile(boot_distances, 97.5, axis=0)
    boot_std = np.std(boot_distances, axis=0)

    # ----------------------------------------------------
    # 4. Pairwise Hypothesis Testing (Comparing Step Sizes)
    # ----------------------------------------------------
    # Build on a raw NumPy array first to avoid Pandas 2.0+ read-only memory issues
    raw_p_arr = np.full((n_groups, n_groups), np.nan)
    np.fill_diagonal(raw_p_arr, 1.0)

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            # Bootstrap difference vector (Step i vs. Step j)
            diff = boot_distances[:, i] - boot_distances[:, j]

            # Non-parametric bootstrap two-tailed p-value
            p_val = 2 * min(
                (np.sum(diff >= 0) + 1) / (num_bootstraps + 1),
                (np.sum(diff <= 0) + 1) / (num_bootstraps + 1)
            )
            raw_p_arr[i, j] = p_val
            raw_p_arr[j, i] = p_val

    # Apply Benjamini-Hochberg FDR correction over the upper triangle comparisons
    triu_idx = np.triu_indices(n_groups, k=1)
    raw_pvals_vector = raw_p_arr[triu_idx]

    _, corrected_pvals, _, _ = multipletests(
        raw_pvals_vector,
        alpha=alpha,
        method='fdr_bh'
    )

    # Construct symmetric corrected p-value matrix
    adj_p_arr = np.ones((n_groups, n_groups))
    adj_p_arr[triu_idx] = corrected_pvals
    adj_p_arr[(triu_idx[1], triu_idx[0])] = corrected_pvals
    np.fill_diagonal(adj_p_arr, 1.0)

    # ----------------------------------------------------
    # 5. Format Outputs as Clean DataFrames
    # ----------------------------------------------------
    step_summary_df = pd.DataFrame({
        "Transition": step_labels,
        "Observed_ND_Distance": obs_distances,
        "SD_Bootstrap": boot_std,
        "CI_95_Lower": ci_lower,
        "CI_95_Upper": ci_upper
    }).set_index("Transition")

    step_comparison_df = pd.DataFrame(
        adj_p_arr,
        index=step_labels,
        columns=step_labels
    )

    # Save data frames to output directory
    step_summary_df.to_csv(output_dir / f'{title}_step_distances_summary.csv')
    step_comparison_df.to_csv(output_dir / f'{title}_step_comparison_p_values.csv')

    # ----------------------------------------------------
    # 6. High-Legibility Visualization Code
    # ----------------------------------------------------
    # sns.set_theme(style="whitegrid", rc={"axes.grid": False})
    fig, ax = plt.subplots(figsize=(6, 6))

    # --- Step Distances Line Plot ---
    x_positions = np.arange(n_groups)
    y_err = [obs_distances - ci_lower, ci_upper - obs_distances]

    ax.errorbar(
        x_positions, obs_distances, yerr=y_err,
        fmt='o-', color='#1f77b4', ecolor='#7f7f7f',
        elinewidth=1.5, capsize=4, markersize=7,
        label='Mean Dist with 95% CI'
    )

    ax.grid(True, which='both', linestyle=':', alpha=0.6, color='#cbcbcb')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.set_ylabel("N-Dimensional Euclidean Distance", fontsize=12, labelpad=10)
    ax.set_xlabel("Transition Step", fontsize=12, labelpad=10)

    # Align Title & Layout
    plt.suptitle(f"Periodic Manifold Geometry Analysis: {title}", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)

    plt.savefig(output_dir / f'{title}_periodic_spacing_analysis.svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Analysis complete! Saved plots and tables to: {output_dir}")
    return step_summary_df, step_comparison_df

def ori_ring_plot(
        ori_pca_data: pd.DataFrame,
        title: str,
        output_dir: Path
):
    #components_to_use = find_max_seperation_dataframe(pca_dataframe=ori_pca_data, num_comps=2)
    components_to_use = [0,1]
    orientation_data = ori_pca_data.index.astype(int).to_numpy()
    output_dir = output_dir / 'ori_ring_plots'
    output_dir.mkdir(parents=True, exist_ok=True)

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

    analyze_periodic_spacing(
        ori_pca_data,output_dir=output_dir, title=title)


def ori_explained_variance_plot(
        explained_variance: list,
        title: str,
        output_dir: Path
):
    pr = participation_ratio(explained_variance)

    plt.figure()
    plt.plot(explained_variance, marker='o', markersize=3, label='Explained Variance')

    plt.title(title)
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance')
    plt.axvline(pr - 1, color='purple', linestyle=':', linewidth=1,
                label=f'PR = {pr:.2f}')
    plt.legend()
    plt.tight_layout()  # Keeps everything neatly contained

    plt.savefig(output_dir / f'{title}_explained_variance.svg')
    plt.savefig(output_dir / f'{title}_explained_variance.png')

def participation_ratio(explained_variance: np.ndarray) -> float:
    """
    Effective dimensionality via participation ratio, computed on the
    full eigenspectrum: PR = (sum(lambda_i))^2 / sum(lambda_i^2).
    Returns a continuous value between 1 and len(explained_variance).
    """
    explained_variance = np.asarray(explained_variance, dtype=float)
    return float(explained_variance.sum() ** 2 / np.sum(explained_variance ** 2))

