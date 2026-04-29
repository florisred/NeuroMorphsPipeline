import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from collections import defaultdict
from utils.utils import scale_session
from data_objects.pca_data import PCAData


def rdm_analysis_full(pca_data_dict: dict[str, PCAData], **kwargs):
    kwargs['full_data'] = True
    rdm_analysis(pca_data_dict, **kwargs)


def rdm_analysis(pca_data_dict: dict[str, PCAData], **kwargs):
    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path):
        raise ValueError('output_dir not provided or is not a Path object')

    rdm_output_dir = output_dir / 'rdm'
    rdm_output_dir.mkdir(parents=True, exist_ok=True)

    n_components = kwargs.get('n_components', 2)
    dist_metric = kwargs.get('dist_metric', 'euclidean')
    full_data = kwargs.get('full_data', False)
    show = kwargs.get('show', False)

    # 1. Separate grouping logic from calculation logic
    analysis_groups = defaultdict(list)

    for t_key, pca_data in pca_data_dict.items():
        pca_data_normalized = pca_data.copy()
        pca_data_normalized.normalize()
        if full_data:
            if 'subset' in t_key: continue
            analysis_groups['full_dataset'].append(pca_data_normalized)
        else:
            if 'full' in t_key: continue
            ds_key = t_key.split('_')[0]
            subset_name = t_key[len(ds_key) + 1:]
            analysis_groups[subset_name].append(pca_data_normalized)

    all_rdms = defaultdict(list)
    stb_mtcs = []
    sbtr_mtcs = defaultdict(list)

    # 2. Process each dataset group independently
    for group_name, pca_data_list in analysis_groups.items():
        if not pca_data_list: continue

        rdms = {}
        names = [d.name for d in pca_data_list]

        # Calculate RDMs via pdist
        for pca_data in pca_data_list:
            data = pca_data.get_data_components(n_components=n_components)
            distance_vector = pdist(data.values, metric=dist_metric)
            rdms[pca_data.name] = distance_vector
            all_rdms[pca_data.data_source].append(distance_vector)

        # Initialize stability matrix for the current group
        n_items = len(names)
        stability_matrix = np.zeros((n_items, n_items))

        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                # Calculate stability (Spearman correlation)
                stability_matrix[i, j], _ = spearmanr(rdms[n1], rdms[n2])

                # Calculate scaled subtractions
                normalized_n1 = scale_session(rdms[n1].reshape(-1, 1))
                normalized_n2 = scale_session(rdms[n2].reshape(-1, 1))

                pair_key = f'{n1.split("_")[0]}-{n2.split("_")[0]}'
                sbtr_mtcs[pair_key].append((normalized_n1 - normalized_n2).flatten())

        # Append only ONCE per group, after the matrix is fully populated
        stb_mtcs.append(stability_matrix)

    # 3. Average and Plot RDMs
    for key, value_list in all_rdms.items():
        if not value_list: continue
        avg_rdm = np.mean(value_list, axis=0)
        sqr = squareform(avg_rdm)
        nms = np.arange(sqr.shape[0])
        _plot_rdm(sqr, nms, f"avg_{key}", rdm_output_dir, show=show)

    # 4. Average and Plot Subtraction Matrices
    for key, value_list in sbtr_mtcs.items():
        if not value_list: continue
        avg_rdm = np.mean(value_list, axis=0)
        sqr = squareform(avg_rdm)
        nms = np.arange(sqr.shape[0])
        _plot_rdm(sqr, nms, f"subtraction_{key}", rdm_output_dir, full_data=full_data, show=show)

    # 5. Average and Plot Stability Matrix
    if stb_mtcs:
        # Note: This np.mean assumes all groups had the exact same number of `names`.
        # If subsets have different lengths, this will raise a numpy broadcast error.
        avg_stb_mx = np.mean(stb_mtcs, axis=0)
        _plot_stability(avg_stb_mx, names, rdm_output_dir, name='stability_avg', show=show, full_data=full_data)


def _plot_stability(matrix, labels, output_dir, name, show, full_data):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='viridis',
                xticklabels=labels, yticklabels=labels, square=True)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    if full_data: name = name + (f"full")
    plt.title(f"Representational Stability ({name})", pad=20, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png")
    if show: plt.show()
    plt.close()  # Critical: Prevent memory leaks


def _plot_rdm(matrix, labels, name, output_dir, show, full_data=False):
    if full_data: name = name + (f"full")
    scale_factor = max(10, len(labels) * 0.3)
    plt.figure(figsize=(scale_factor, scale_factor))
    ax = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels,
                     cmap='rocket', square=True)

    if len(labels) > 20:
        n = len(labels) // 20
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            if i % n != 0: label.set_visible(False)
        for i, label in enumerate(ax.yaxis.get_ticklabels()):
            if i % n != 0: label.set_visible(False)

    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(f"RDM: {name}", pad=20, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_dir / f"rdm_{name}.png", bbox_inches='tight')
    if show: plt.show()
    plt.close()  # Critical: Prevent memory leaks