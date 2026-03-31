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

def rdm_analysis(
        pca_data_dict: dict[str, PCAData], **kwargs
):
    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path): raise ValueError('output_dir not provided')
    rdm_output_dir = output_dir / 'rdm'
    rdm_output_dir.mkdir(parents=True, exist_ok=True)
    n_components = kwargs.get('n_components', 2)
    dist_metric = kwargs.get('dist_metric', 'euclidean')
    avg_only = kwargs.get('avg_only', True)
    full_data = kwargs.get('full_data', False)
    show = kwargs.get('show', False)


    subset_groups = defaultdict(list)
    all_rdms = defaultdict(list)
    stb_mtcs = []
    sbtr_mtcs = defaultdict(list)
    for t_key in pca_data_dict.keys():
        if full_data:
            if 'subset'in t_key: continue
        else:
            if 'full' in t_key: continue
        ds_key = t_key.split('_')[0]
        subset_name = t_key[len(ds_key) + 1:]

    if subset_name not in subset_groups:
        subset_groups[subset_name].append(pca_data_dict[t_key])
    for subset_name, pca_data_list in subset_groups.items():
        if len(pca_data_list) > 0:
            rdm_output_dir.mkdir(parents=True, exist_ok=True)
            rdms = {}
            names = [d.name for d in pca_data_list]
            for pca_data in pca_data_list:
                data = pca_data.get_data_components(n_components=n_components)
                distance_vector = pdist(data.values, metric=dist_metric)
                rdms[pca_data.name] = distance_vector
                all_rdms[pca_data.data_source].append(distance_vector)
                morph_names = pca_data.metadata.morph_names
            stability_matrix = np.zeros((len(names), len(names)))
            for i, n1 in enumerate(names):
                for j, n2 in enumerate(names):
                    stability_matrix[i, j], _ = spearmanr(rdms[n1], rdms[n2])
                    normalized_n1 = scale_session(rdms[n1].reshape(-1, 1))
                    normalized_n2 = scale_session(rdms[n2].reshape(-1, 1))
                    sbtr_mtcs[f'{n1.split("_")[0]}-{n2.split("_")[0]}'].append((normalized_n1 - normalized_n2).flatten())
            stb_mtcs.append(stability_matrix)

            if not avg_only:
                _plot_stability(stability_matrix, names, rdm_output_dir, name='what', show=show)
                for name, dist_vec in rdms.items():
                    _plot_rdm(squareform(dist_vec), morph_names, name, rdm_output_dir, show=show)
                for name, dist_vec in sbtr_mtcs.items():
                    _plot_rdm(squareform(dist_vec), morph_names, name, rdm_output_dir, show=show)

    for key, value in all_rdms.items():
        avg_rdm = np.mean(value, axis=0)
        sqr = squareform(avg_rdm)
        nms = np.arange(sqr.shape[0])
        _plot_rdm(sqr, nms, f"avg_{key}", rdm_output_dir, show=show)
    for key, value in sbtr_mtcs.items():
        avg_rdm = np.mean(value, axis=0)
        sqr = squareform(avg_rdm)
        nms = np.arange(sqr.shape[0])
        _plot_rdm(sqr, nms, f"subtraction_{key}", rdm_output_dir, show=show)

    avg_stb_mx = np.mean(stb_mtcs, axis=0)
    _plot_stability(avg_stb_mx, all_rdms.keys(), rdm_output_dir, name='stability_avg', show=show)

def _plot_stability(matrix, labels, output_dir, name, show):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='viridis',
                xticklabels=labels, yticklabels=labels, square=True)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(f"Representational Stability {name})", pad=20, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png")
    if show: plt.show()

def _plot_rdm(matrix, labels, name, output_dir, show):
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
    if show: plt.show()  # Forces the plot to show in the loop