import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict
from utils.utils import scale_session
from data_objects.pca_data import PCAData
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rdr
import rsatoolbox.vis as rdv
import rsatoolbox.data.noise as rsn

def rdm_analysis_anchor(
        pca_data_dict: dict[str, PCAData],
        **kwargs
):
    pca_data_dict_filtered = {}
    for name, pca_data in pca_data_dict.items():
        if 'anchor' in name:
            pca_data_dict_filtered[name] = pca_data
    kwargs['rdm_type'] = 'anchor'
    kwargs['output_dir'] = kwargs.get('output_dir') / 'rdm_cross_anchor'

    _rdm_analysis(pca_data_dict_filtered, **kwargs)


def rdm_analysis_full(pca_data_dict: dict[str, PCAData], **kwargs):
    pca_data_dict_filtered = {}
    for name, pca_data in pca_data_dict.items():
        if 'full' in name:
            pca_data_dict_filtered[name] = pca_data
    kwargs['rdm_type'] = 'full'
    kwargs['output_dir'] = kwargs.get('output_dir') / 'rdm_cross_full'
    _rdm_analysis(pca_data_dict_filtered, **kwargs)

def rdm_analysis_ori(pca_data_dict: dict[str, PCAData], **kwargs):
    kwargs['rdm_type'] = 'full'
    kwargs['output_dir'] = kwargs.get('output_dir') / 'rdm_cross_full'
    kwargs['ori'] = True
    _rdm_analysis(pca_data_dict, **kwargs)

def rdm_analysis_subsets(pca_data_dict: dict[str, PCAData], **kwargs):
    output_dir = kwargs.get('output_dir') / 'rdm_cross_subsets'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by cycle name, across datasources
    # keys look like: 'TwoPhoton_honeycomb__sand__turtoise'
    # we want to group by 'honeycomb__sand__turtoise'
    cycles = defaultdict(dict)
    for key, pca_data in pca_data_dict.items():
        if 'subset' not in key: continue
        data_source = pca_data.data_source
        cycle_name = key[len(data_source) + 1:]  # strip datasource prefix
        cycles[cycle_name][data_source] = pca_data

    all_stability_matrices = []
    stability_names = None

    for cycle_name, source_dict in cycles.items():
        cycle_output_dir = output_dir / cycle_name
        cycle_output_dir.mkdir(parents=True, exist_ok=True)

        rdm_vectors = {}

        for data_source, pca_data in source_dict.items():
            sorted_arr = pca_data.metadata.morph_names.drop_duplicates().to_numpy()
            numpy_data = pca_data.raw_data.to_numpy()
            names = pca_data.raw_data.index.to_numpy()
            session_labels, _ = make_session_labels(names)

            obs_descriptors = {
                'conds': names,
                'sessions': session_labels
            }


            if data_source in ['TwoPhoton', 'GaborNet', 'RetinodivnormGaborNet']:
                dataset = rsd.Dataset(
                    measurements=numpy_data,
                    obs_descriptors=obs_descriptors
                )
                noise_precision = rsn.prec_from_unbalanced(
                    dataset,
                    obs_desc='conds',
                    method='shrinkage_diag'
                )
                rdm_cv = rdr.calc_rdm(
                    dataset,
                    method='crossnobis',
                    descriptor='conds',
                    cv_descriptor='sessions',
                    noise=noise_precision
                )
            else:
                dataset = rsd.Dataset(
                    measurements=scale_session(numpy_data),
                    obs_descriptors=obs_descriptors
                )
                rdm_cv = rdr.calc_rdm(
                    dataset,
                    method='euclidean',
                    descriptor='conds',
                )

            # Reorder to match sorted_arr
            unsorted_labels = rdm_cv.pattern_descriptors['conds']
            name_to_idx = {name: i for i, name in enumerate(unsorted_labels)}
            reorder_indices = np.array([
                name_to_idx[name] for name in sorted_arr
                if name in name_to_idx
            ])
            rdm_cv.reorder(reorder_indices)

            # Plot individual RDM for this cycle + datasource
            rdv.show_rdm(
                rdm_cv,
                show_colorbar='figure',
                pattern_descriptor='conds',
                figsize=(20, 15),
                cmap='rocket'
            )
            plt.title(f"{data_source} - {cycle_name}")
            plt.savefig(cycle_output_dir / f"{data_source}.png")
            plt.close()

            rdm_vectors[data_source] = rdm_cv.get_vectors()[0]

        # Stability matrix for this cycle
        source_names = list(rdm_vectors.keys())
        n = len(source_names)
        stability_matrix = np.zeros((n, n))

        for i, n1 in enumerate(source_names):
            for j, n2 in enumerate(source_names):
                r, _ = spearmanr(rdm_vectors[n1], rdm_vectors[n2])
                stability_matrix[i, j] = r

        _plot_stability(
            stability_matrix,
            source_names,
            cycle_output_dir,
            name=f'Stability_{cycle_name}',
            show=False,
            full_data=False
        )

        all_stability_matrices.append(stability_matrix)
        stability_names = source_names  # same across all cycles

    # Average stability matrix across all 24 cycles
    if all_stability_matrices:
        safe_matrices = np.clip(all_stability_matrices, -0.9999, 0.9999)
        z_matrices = np.arctanh(safe_matrices)
        avg_z_matrix = np.mean(z_matrices, axis=0)
        avg_matrix = np.tanh(avg_z_matrix)
        _plot_stability(
            avg_matrix,
            stability_names,
            output_dir,
            name='Stability_average_across_cycles',
            show=True,
            full_data=False
        )

def _rdm_analysis(
        pca_data_dict: dict[str, PCAData],
        **kwargs
):
    rdms = []
    rdm_vectors = {}  # store as dict of name -> distance vector
    output_dir = kwargs.get('output_dir')
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, pca_data in pca_data_dict.items():
        data_source = pca_data.data_source
        sorted_arr = pca_data.metadata.morph_names.drop_duplicates().to_numpy()
        numpy_data = pca_data.raw_data.to_numpy()
        names = pca_data.raw_data.index.to_numpy()
        session_labels, min_trials = make_session_labels(names)

        obs_descriptors = {
            'conds': names,
            'sessions': session_labels  # <-- store the array here
        }


        if data_source in ['TwoPhoton', 'GaborNet', 'RetinodivnormGaborNet']:
            dataset = rsd.Dataset(
                measurements=numpy_data,
                obs_descriptors=obs_descriptors
            )
            noise_precision = rsn.prec_from_unbalanced(
                dataset,
                obs_desc='conds',
                method='shrinkage_diag'
            )
            rdm_cv = rdr.calc_rdm(
                dataset,
                method='crossnobis',
                descriptor='conds',
                cv_descriptor='sessions',
                noise=noise_precision
            )
        else:
            dataset = rsd.Dataset(
                measurements=scale_session(numpy_data),
                obs_descriptors=obs_descriptors
            )
            rdm_cv = rdr.calc_rdm(
                dataset,
                method='euclidean',
                descriptor='conds',
            )
        if kwargs.get('ori', False) is False:
            unsorted_labels = rdm_cv.pattern_descriptors['conds']
            desired = sorted_arr
            name_to_idx = {name: i for i, name in enumerate(unsorted_labels)}
            reorder_indices = np.array([name_to_idx[name] for name in desired])
            rdm_cv.reorder(reorder_indices)

        rdm_cv.rdm_descriptors['name'] = [data_source]
        rdms.append(rdm_cv)
        rdm_vectors[key] = rdm_cv.get_vectors()[0]  # store the flat vector

        rdv.show_rdm(
            rdm_cv,
            show_colorbar='figure',
            pattern_descriptor='conds',
            rdm_descriptor=data_source,
            figsize=(20,30),
            cmap='rocket'
        )
        plt.savefig(output_dir / f"{key}.png")
        plt.close()

    names = list(rdm_vectors.keys())
    n = len(names)
    comparison_matrix = np.zeros((n, n))

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            r, _ = spearmanr(rdm_vectors[n1], rdm_vectors[n2])
            comparison_matrix[i, j] = r

    _plot_stability(
        comparison_matrix,
        names,
        output_dir,
        name=f'Representational Stability {kwargs.get("rdm_type")}',
        show=True,
        full_data=True
    )



def make_session_labels(cond_labels: np.ndarray) -> np.ndarray:
    """
    Assigns session labels 0..n_trials-1 within each condition.
    For conditions with fewer trials than the max, cycles the labels
    so every condition has the same number of unique session values.
    """
    unique_conds, counts = np.unique(cond_labels, return_counts=True)
    min_trials = counts.min()

    session_labels = np.zeros(len(cond_labels), dtype=int)

    for cond in unique_conds:
        mask = cond_labels == cond
        n = mask.sum()
        indices = np.where(mask)[0]
        session_labels[indices] = np.arange(n) % min_trials

    return session_labels, min_trials


def _plot_stability(matrix, labels, output_dir, name, show, full_data):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='viridis',
                xticklabels=labels, yticklabels=labels, square=True)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(f"{name}", pad=20, fontsize=15)
    plt.tight_layout()
    if full_data: name = name + (f"full")
    plt.savefig(output_dir / f"{name}.png")
    if show: plt.show()
    plt.close()  # Critical: Prevent memory leaks
