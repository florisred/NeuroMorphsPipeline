from collections import defaultdict

from data_objects.pca_data import PCAData
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from utils.utils import find_max_separation
from utils.utils import make_one_equal_two
import pandas as pd
import seaborn as sns


def create_subset_plots(pca_data_dict: dict[str, PCAData], with_variability=False, **kwargs):
    calculate_deviation(pca_data_dict, source='TwoPhoton')
    components = find_max_separation(pca_data_dict=pca_data_dict, num_comps=2)
    output_dir = kwargs.get('output_dir')
    curve_metrics_only = kwargs.get('curve_metrics_only', False)
    absolute = kwargs.get('absolute', False)

    if not output_dir or not isinstance(output_dir, Path):
        raise ValueError('output_dir not provided or not a Path object')
    output_dir = output_dir / 'subsets'
    output_dir.mkdir(parents=True, exist_ok=True)
    ppd_dict = {}
    pair_key_curvatures = defaultdict(dict)
    for key, pca_data in pca_data_dict.items():
        if 'subset' not in key:
            continue
        data_source_name = pca_data.data_source
        normalized_pca_data = pca_data.normalize()
        if normalized_pca_data is None: raise ValueError('Expected PCAData object, but got NoneType instead')
        folded_ppd, pair_key_curves = None, calculate_curvature(pca_data, components = components[key])
        for pair_key, curve_metric in pair_key_curves.items():
            ds_pair_data = pair_key_curvatures[data_source_name].get(pair_key, None)
            if ds_pair_data is None:
                pair_key_curvatures[data_source_name][pair_key] = curve_metric
            else:
                combined_array = np.column_stack([ds_pair_data, curve_metric])
                pair_key_curvatures[data_source_name][pair_key] = combined_array


        if not curve_metrics_only:
            pc_x, pc_y = components[key][0], components[key][1]
            data = pca_data.pca_data
            metadata = pca_data.metadata
            numeric_index = pca_data.get_numeric_index()
            plt.figure(figsize=(12.5, 7.5))
            ax = plt.gca()
            plot_coords = data.iloc[:, [pc_x, pc_y]].values
            loop_data = np.vstack([plot_coords, plot_coords[0]])
            plt.plot(loop_data[:, 0], loop_data[:, 1],
                     color='gray', alpha=0.4, linestyle='--', zorder=1, label='Morph Path')
            if with_variability and hasattr(pca_data, 'trial_data') and pca_data.trial_data is not None:
                cmap = plt.get_cmap('viridis')
                norm = plt.Normalize(vmin=np.min(numeric_index), vmax=np.max(numeric_index))
                for i, name in enumerate(metadata.morph_names):
                    try:
                        morph_trials = pca_data.trial_data.loc[name]
                        if isinstance(morph_trials, pd.Series):
                            trial_coords = morph_trials.iloc[[pc_x, pc_y]].values.reshape(1, -1)
                        else:
                            trial_coords = morph_trials.iloc[:, [pc_x, pc_y]].values
                        color = cmap(norm(numeric_index[i]))
                        plt.scatter(trial_coords[:, 0], trial_coords[:, 1],
                                    color=color, s=20, alpha=0.3, zorder=1, edgecolors='none')
                        if len(trial_coords) >= 2:
                            cov = np.cov(trial_coords[:, 0], trial_coords[:, 1])
                            if cov.ndim == 2:
                                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                                order = eigenvalues.argsort()[::-1]
                                eigenvalues = eigenvalues[order]
                                eigenvectors = eigenvectors[:, order]
                                angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                                n_std = 1
                                width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 0))
                                ell = Ellipse(xy=(np.mean(trial_coords[:, 0]), np.mean(trial_coords[:, 1])),
                                              width=width, height=height, angle=angle,
                                              facecolor=color, alpha=0.15, edgecolor=color,
                                              linewidth=1.5, zorder=1)
                                ax.add_patch(ell)
                    except KeyError:
                        pass
            plt.scatter(plot_coords[:, 0], plot_coords[:, 1],
                        c=numeric_index, cmap='viridis', s=60, alpha=0.8,
                        edgecolors='white', zorder=2)
            is_anchor = metadata.anchor_mask
            for i in np.where(is_anchor)[0]:
                name = metadata.morph_names.iloc[i]
                plt.text(plot_coords[i, 0], plot_coords[i, 1] + 0.5, name, fontsize=10,
                         fontweight='bold', ha='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))
            cbar = plt.colorbar()
            cbar.set_ticks(numeric_index)
            cbar.set_ticklabels(metadata.morph_names)
            plt.xlabel(f'PC{pc_x + 1}')
            plt.ylabel(f'PC{pc_y + 1}')
            plt.title(f'{pca_data.name} 2D (PC{pc_x + 1} vs PC{pc_y + 1})')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.savefig(output_dir / f'{pca_data.name}.png')
            plt.close()

    plt.figure(figsize=(12.5, 7.5))
    # for data_source_name, ppd in ppd_dict.items():
    #     per_point_deviation = np.mean(ppd, axis=0)
    #     std_per_point_deviation = np.std(ppd, axis=0)
    #     plt.plot(per_point_deviation, label=data_source_name)
    #     plt.fill_between(
    #         range(len(per_point_deviation)),
    #         per_point_deviation - std_per_point_deviation,
    #         per_point_deviation + std_per_point_deviation,
    #         color='green', alpha=0.2
    #     )
        ## ToDO: add correlation between pair_keys!
    # plt.legend()
    # plt.savefig(output_dir / 'mean_curves.png')
    # plt.show()

    plt.figure(figsize=(12.5, 7.5))
    baseline_source = kwargs.get('baseline_source', 'TwoPhoton')
    baseline_deviation = np.mean(ppd_dict[baseline_source], axis=0)
    for data_source_name, ppd in ppd_dict.items():
        if data_source_name == baseline_source: continue
        per_point_deviation = np.mean(ppd, axis=0)
        residuals = baseline_deviation - per_point_deviation
        if absolute:
            residuals = np.abs(residuals)
        mean_residuals = np.mean(residuals)
        plt.bar(data_source_name, mean_residuals)
    plt.title('mean difference from 2p')
    plt.xticks(rotation=45)
    plt.show()

    for data_source_name, pair_key_values in pair_key_curvatures.items():
        # Pro-tip: Convert your dict to a long-form DataFrame for easier plotting
        plot_data = []
        for pair_key, arr in pair_key_values.items():
            # Assuming the last column is your curvature metric
            curvatures = arr[:, -1]
            for step, val in enumerate(curvatures):
                source = pair_key.split('__')[0]
                plot_data.append({'step': step, 'curvature': val, 'pair': pair_key, 'source': source})

    df_plot = pd.DataFrame(plot_data)

    # Use Seaborn's FacetGrid
    g = sns.relplot(
        data=df_plot, x="step", y="curvature",
        col="source", hue="pair", kind="line",
        col_wrap=3, height=3, facet_kws={'sharey': False}
    )
    plt.show()

    heatmap_df = df_plot.pivot(index="pair", columns="step", values="curvature")

    plt.figure(figsize=(10, 12))
    sns.heatmap(heatmap_df, annot=False, cmap="viridis")
    plt.title("Curvature Progression per Pair")
    plt.show()
    test=1


def calculate_curvature(pca_data: PCAData, components: tuple[int, int]):
    data = pca_data.pca_data.iloc[:, list(components)]
    anchors = pca_data.anchors.iloc[:, list(components)]
    anchors_unique = anchors.drop_duplicates()
    centre = np.mean(anchors_unique, axis=0)
    metadata = pca_data.metadata
    n_subsets = pca_data.n_unique_anchors

    pair_key_curvatures = {}
    for pair_key in pca_data.metadata.get_pair_keys(unique=True):
        ideal_data = []
        pair_key_mask = metadata.get_pair_keys(unique=False, dropna=False) == pair_key
        pair_key_data = data[pair_key_mask]
        src_cat = np.unique(metadata.get_metadata()['src_cat'][pair_key_mask])
        dst_cat = np.unique(metadata.get_metadata()['dst_cat'][pair_key_mask])
        if len(src_cat) != 1 or len(dst_cat) != 1:
            raise ValueError(f'Multiple src_cat or dst_cat values for pair_key {pair_key}')
        src_cat, dst_cat = src_cat[0], dst_cat[0]
        p0 = anchors_unique.loc[src_cat]
        p1 = anchors_unique.loc[dst_cat]
        vector = p1 - p0
        norm_steps = metadata.morph_steps.loc[pair_key_mask]
        ideal_data.append(p0)
        for norm_step in norm_steps:
            ideal_data.append(p0 + norm_step * vector)
        ideal_data.append(p1)

        ideal_df = pd.DataFrame(ideal_data)
        closer_array = np.where(np.linalg.norm(ideal_df - centre, axis=1) > np.linalg.norm(pair_key_data - centre, axis=1), -1,
                                1)
        residuals = ideal_df - data
        distances = np.linalg.norm(residuals, axis=1)
        distances_adjusted = distances * closer_array

        pair_key_curvatures[pair_key] = distances_adjusted



    return pair_key_curvatures

def calculate_deviation(pca_data_dict: dict[str, PCAData], source='TwoPhoton'):

    devs = defaultdict(list)
    for key, pca_data in pca_data_dict.items():
        data_source = pca_data.data_source
        subset = key[len(data_source):]
        baseline = pca_data_dict[source+subset].pca_data
        devs[data_source].append(np.linalg.norm(baseline -pca_data.pca_data, axis=1))
    print(devs)
    test=1




"""
        

        p0 = anchors_unique.loc[src_cat].values
        p1 = anchors_unique.loc[dst_cat].values
        line_vec = p1 - p0
        L2_sq = np.sum(line_vec ** 2)
        L = np.sqrt(L2_sq)
        d = L / 2
        points = data.loc[pair_key_mask, [0, 1]].values  # Assuming cols 0 and 1
        w = points - p0  # Vector from p0 to p3s
        a = np.dot(w, line_vec) / L
        b = (line_vec[0] * w[:, 1] - line_vec[1] * w[:, 0]) / L
        denominator = (a - d) ** 2 - d ** 2
        curve_metric = np.divide(b, denominator, out=np.zeros_like(b), where=denominator != 0)
        curve_metric = np.abs(curve_metric)
        pair_key_curvatures[pair_key] = ppd

"""





