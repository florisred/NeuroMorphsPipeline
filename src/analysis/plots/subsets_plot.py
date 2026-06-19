from collections import defaultdict

from data_objects.pca_data import PCAData
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from utils.utils import find_max_separation
import pandas as pd
import seaborn as sns


def create_subset_plots(pca_data_dict: dict[str, PCAData], with_variability=False, **kwargs):
    components = find_max_separation(pca_data_dict=pca_data_dict, num_comps=2, filter='subset')
    output_dir = kwargs.get('output_dir')

    if not output_dir or not isinstance(output_dir, Path):
        raise ValueError('output_dir not provided or not a Path object')
    output_dir = output_dir / 'subsets'
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_key_curvatures = defaultdict(dict)
    for key, pca_data in pca_data_dict.items():
        if 'subset' not in key:
            continue
        data_source_name = pca_data.data_source
        pca_data = pca_data.normalize()
        pair_key_curves = calculate_curvature(pca_data, components = components[key])
        for pair_key, curve_metric in pair_key_curves.items():
            ds_pair_data = pair_key_curvatures[data_source_name].get(pair_key, None)
            if ds_pair_data is None:
                pair_key_curvatures[data_source_name][pair_key] = curve_metric
            else:
                combined_array = np.column_stack([ds_pair_data, curve_metric])
                pair_key_curvatures[data_source_name][pair_key] = combined_array



        pc_x, pc_y = components[key][0], components[key][1]
        data = pca_data.pca_data
        metadata = pca_data.metadata
        numeric_index = pca_data.get_numeric_index()
        plt.figure(figsize=(12.5, 7.5))
        ax = plt.gca()
        plot_coords = data.iloc[:, [pc_x, pc_y]].values
        loop_data = np.vstack([plot_coords, plot_coords[0]])
        plt.plot(loop_data[:, 0], loop_data[:, 1],
                 color='gray', alpha=0.4, linestyle='-', zorder=1, label='Morph Path')

        is_anchor = metadata.anchor_mask.values
        anchor_coords = plot_coords[is_anchor]

        if len(anchor_coords) >= 3:
            ideal_loop = np.vstack([anchor_coords, anchor_coords[0]])
            plt.plot(ideal_loop[:, 0], ideal_loop[:, 1],
                     color='gray', alpha=0.5, linestyle='--',
                     linewidth=1.5, zorder=1, label='Ideal Path')


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


        ## ToDO: add correlation between pair_keys!

    curv_output_dir = output_dir / 'curvature_metrics'
    curv_output_dir.mkdir(exist_ok=True)
    
    for data_source_name, pair_key_values in pair_key_curvatures.items():
        plot_data = []
        for pair_key, arr in pair_key_values.items():
            curvatures = arr[:, -1]
            for step, val in enumerate(curvatures):
                source = pair_key.split('__')[0]
                plot_data.append({'step': step, 'deviation': val, 'pair': pair_key, 'source': source})

        df_plot = pd.DataFrame(plot_data)
        df_plot.to_csv(curv_output_dir / f'deviation_data_{data_source_name}.csv')

        # Use Seaborn's FacetGrid
        g = sns.relplot(
            data=df_plot, x="step", y="deviation",
            col="source", hue="pair", kind="line",
            col_wrap=3, height=3, facet_kws={'sharey': False}
        )
        plt.title(f'{data_source_name}')
        plt.savefig(curv_output_dir / f'deviation_data_{data_source_name}.png')

        heatmap_df = df_plot.pivot(index="pair", columns="step", values="deviation")

        plt.figure(figsize=(10, 12))
        sns.heatmap(heatmap_df, annot=False, cmap="viridis")
        plt.title(f"Curvature Progression per Pair, {data_source_name}")
        plt.savefig(curv_output_dir / f'curvature_progression_{data_source_name}.png')



def calculate_curvature(pca_data: PCAData, components: tuple[int, int]):
    data = pca_data.pca_data.iloc[:, list(components)]
    anchors = pca_data.anchors.iloc[:, list(components)]
    anchors_unique = anchors.drop_duplicates()
    metadata = pca_data.metadata

    centre = np.mean(anchors_unique.values, axis=0)

    pair_key_curvatures = {}
    unique_pairs = metadata.get_pair_keys(unique=True)

    for pair_key in unique_pairs:
        # Get mask and data for this specific morph pair
        pair_key_mask = metadata.get_pair_keys(unique=False, dropna=False) == pair_key
        pair_key_data = data[pair_key_mask]

        src_cat = np.unique(metadata.get_metadata()['src_cat'][pair_key_mask])
        dst_cat = np.unique(metadata.get_metadata()['dst_cat'][pair_key_mask])

        p0 = anchors_unique.loc[src_cat[0]].values
        p1 = anchors_unique.loc[dst_cat[0]].values

        line_vec = p1 - p0
        line_length = np.linalg.norm(line_vec)
        line_unit = line_vec / line_length if line_length > 0 else line_vec

        norm_steps = metadata.morph_steps.loc[pair_key_mask].values
        ideal_points = np.array([p0 + step * line_vec for step in norm_steps])

        perp_vec = np.array([-line_unit[1], line_unit[0]])

        midpoint = (p0 + p1) / 2
        outward_dir = midpoint - centre


        if np.dot(perp_vec, outward_dir) < 0:
            perp_vec = -perp_vec

        # 5. Calculate signed distance from the ideal line
        # residuals represent the vector from the 'ideal' point to the 'actual' point
        residuals = pair_key_data.values - ideal_points

        # Project residuals onto our oriented perp_vec
        # Positive = expanded (away from centre), Negative = contracted (towards centre)
        signed_distances = residuals @ perp_vec

        # 6. Pad with zeros for the anchors (Step 0 and Step N)
        # Assuming your heatmap expects the full sequence including the fixed ends
        distances_final = np.pad(signed_distances, (1, 1), constant_values=0)
        pair_key_curvatures[pair_key] = distances_final

    return pair_key_curvatures




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





