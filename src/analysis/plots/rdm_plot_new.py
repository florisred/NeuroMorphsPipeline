import numpy as np
import pandas as pd
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
import pickle as pkl
from statsmodels.stats.multitest import multipletests

def fdr_correct_symmetric_matrix(p_matrix: np.ndarray, alpha: float = 0.05):
    """
    Applies Benjamini-Hochberg FDR correction to the unique off-diagonal
    entries of a symmetric p-value matrix (e.g. a pairwise correlation
    significance matrix), and returns a symmetric corrected matrix plus a
    symmetric boolean reject matrix. Diagonal entries are left at 1.0 / False,
    since they represent self-comparisons and carry no meaningful p-value.
    """
    n = p_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    raw_pvals = p_matrix[iu]

    corrected_matrix = np.ones((n, n))
    reject_matrix = np.zeros((n, n), dtype=bool)

    if len(raw_pvals) == 0:
        return corrected_matrix, reject_matrix

    reject, corrected, _, _ = multipletests(raw_pvals, alpha=alpha, method='fdr_bh')

    corrected_matrix[iu] = corrected
    corrected_matrix[(iu[1], iu[0])] = corrected
    reject_matrix[iu] = reject
    reject_matrix[(iu[1], iu[0])] = reject
    np.fill_diagonal(corrected_matrix, 1.0)

    return corrected_matrix, reject_matrix

def condition_bootstrap_rdm_difference(mat_ref, mat_m1, mat_m2, n_boot=1000):
    """
    Performs condition-level bootstrapping to compare the representational
    similarity of two models against a biological reference (TwoPhoton) RDM.
    """
    if mat_ref.shape != mat_m1.shape or mat_ref.shape != mat_m2.shape:
        print(f"  [Warning] Matrix shapes do not match. Skipping bootstrap for this pair.")
        print(f"  Shapes: Ref {mat_ref.shape} | M1 {mat_m1.shape} | M2 {mat_m2.shape}")
        return None

    n_conds = mat_ref.shape[0]
    diffs = []

    for _ in range(n_boot):
        # Resample stimulus conditions with replacement
        indices = np.random.choice(n_conds, n_conds, replace=True)

        # Resample the square RDM matrices
        b_ref = mat_ref[indices][:, indices]
        b_m1 = mat_m1[indices][:, indices]
        b_m2 = mat_m2[indices][:, indices]

        # Pull upper triangle indices (excluding diagonal)
        iupper = np.triu_indices(n_conds, k=1)
        vec_ref = b_ref[iupper]
        vec_m1 = b_m1[iupper]
        vec_m2 = b_m2[iupper]

        # Calculate bootstrap Spearman correlations
        r1, _ = spearmanr(vec_ref, vec_m1)
        r2, _ = spearmanr(vec_ref, vec_m2)

        if not np.isnan(r1) and not np.isnan(r2):
            diffs.append(r1 - r2)

    if len(diffs) == 0:
        return None

    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)

    # Calculate two-tailed p-value
    if mean_diff > 0:
        p_val = np.mean(diffs <= 0) * 2
    else:
        p_val = np.mean(diffs >= 0) * 2

    p_val = min(p_val, 1.0)

    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    return mean_diff, ci_lower, ci_upper, p_val


def plot_bootstrap_comparisons(bootstrap_results, output_dir, rdm_type=""):
    """
    Plots the bootstrap comparison results with horizontal error bars representing
    the 95% confidence intervals of the differences in correlation.

    Expects each entry to be a 6-tuple:
    (mean_diff, ci_lower, ci_upper, raw_p, fdr_corrected_p, reject_at_alpha).
    Significance coloring uses the FDR-corrected p-value.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    if not bootstrap_results:
        return

    labels = list(bootstrap_results.keys())
    means = [res[0] for res in bootstrap_results.values()]
    ci_lowers = [res[1] for res in bootstrap_results.values()]
    ci_uppers = [res[2] for res in bootstrap_results.values()]
    raw_p_values = [res[3] for res in bootstrap_results.values()]
    corrected_p_values = [res[4] for res in bootstrap_results.values()]

    errors_left = [m - l for m, l in zip(means, ci_lowers)]
    errors_right = [u - m for m, u in zip(means, ci_uppers)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_positions = np.arange(len(labels))

    colors = []
    for p, m in zip(corrected_p_values, means):
        if p < 0.05:
            colors.append('#1f77b4' if m > 0 else '#d62728')
        else:
            colors.append('#7f7f7f')

    ax.axvline(0, color='#333333', linestyle='--', linewidth=1.2, alpha=0.7, zorder=0)

    for i in range(len(labels)):
        xerr_individual = [[errors_left[i]], [errors_right[i]]]
        ax.errorbar(means[i], y_positions[i], xerr=xerr_individual, fmt='none',
                     ecolor=colors[i], elinewidth=2.5, capsize=5, capthick=1.5, zorder=1)
        ax.plot(means[i], y_positions[i], 'o', color=colors[i],
                markersize=8, markeredgecolor='black', zorder=2)

    formatted_labels = []
    for label, p_raw, p_corr in zip(labels, raw_p_values, corrected_p_values):
        p_raw_text = "< .001" if p_raw < 0.001 else f"{p_raw:.4f}"
        p_corr_text = "< .001" if p_corr < 0.001 else f"{p_corr:.4f}"
        formatted_labels.append(f"{label}\n(raw p = {p_raw_text}, FDR p = {p_corr_text})")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(formatted_labels, fontsize=10)
    ax.set_xlabel("Difference in Spearman Correlation (Δr)\n[Ref vs Model 1] - [Ref vs Model 2]",
                  fontsize=11, fontweight='semibold')
    ax.set_title(f"Bootstrap Model Comparisons ({rdm_type.upper()})\n"
                 f"Relative to TwoPhoton Baseline (95% CI, FDR-BH corrected)", fontsize=12, pad=15)

    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"bootstrap_comparison_{rdm_type}.svg", bbox_inches='tight')
    plt.savefig(output_dir / f"bootstrap_comparison_{rdm_type}.png", dpi=300, bbox_inches='tight')
    plt.close()

def interaction_bootstrap_rdm_difference(
        mat_ref_full, mat_m1_full, mat_m2_full,
        mat_ref_anchor, mat_m1_anchor, mat_m2_anchor,
        n_boot=1000
):
    """
    Performs condition-level bootstrapping to test the interaction effect:
    Does the drop in representational similarity from 'full' to 'anchor'
    differ significantly between Model 1 and Model 2?

    This function handles 'full' and 'anchor' datasets with different dimensions
    by drawing independent bootstrap samples for each set.
    """
    # 1. Verify shapes match within 'full' matrices
    if mat_ref_full.shape != mat_m1_full.shape or mat_ref_full.shape != mat_m2_full.shape:
        print("  [Warning] Full matrices shapes do not match. Skipping bootstrap.")
        return None

    # 2. Verify shapes match within 'anchor' matrices
    if mat_ref_anchor.shape != mat_m1_anchor.shape or mat_ref_anchor.shape != mat_m2_anchor.shape:
        print("  [Warning] Anchor matrices shapes do not match. Skipping bootstrap.")
        return None

    n_conds_full = mat_ref_full.shape[0]
    n_conds_anchor = mat_ref_anchor.shape[0]
    interaction_diffs = []

    for _ in range(n_boot):
        # --- Resample Full Conditions ---
        indices_f = np.random.choice(n_conds_full, n_conds_full, replace=True)
        b_ref_f = mat_ref_full[indices_f][:, indices_f]
        b_m1_f = mat_m1_full[indices_f][:, indices_f]
        b_m2_f = mat_m2_full[indices_f][:, indices_f]

        iupper_f = np.triu_indices(n_conds_full, k=1)
        r_m1_f, _ = spearmanr(b_ref_f[iupper_f], b_m1_f[iupper_f])
        r_m2_f, _ = spearmanr(b_ref_f[iupper_f], b_m2_f[iupper_f])

        # --- Resample Anchor Conditions ---
        indices_a = np.random.choice(n_conds_anchor, n_conds_anchor, replace=True)
        b_ref_a = mat_ref_anchor[indices_a][:, indices_a]
        b_m1_a = mat_m1_anchor[indices_a][:, indices_a]
        b_m2_a = mat_m2_anchor[indices_a][:, indices_a]

        iupper_a = np.triu_indices(n_conds_anchor, k=1)
        r_m1_a, _ = spearmanr(b_ref_a[iupper_a], b_m1_a[iupper_a])
        r_m2_a, _ = spearmanr(b_ref_a[iupper_a], b_m2_a[iupper_a])

        # --- Compute Double Difference ---
        if not (np.isnan(r_m1_f) or np.isnan(r_m2_f) or np.isnan(r_m1_a) or np.isnan(r_m2_a)):
            drop_m1 = r_m1_f - r_m1_a
            drop_m2 = r_m2_f - r_m2_a
            interaction_diffs.append(drop_m1 - drop_m2)

    if len(interaction_diffs) == 0:
        return None

    interaction_diffs = np.array(interaction_diffs)
    mean_interaction = np.mean(interaction_diffs)

    # Calculate two-tailed p-value
    if mean_interaction > 0:
        p_val = np.mean(interaction_diffs <= 0) * 2
    else:
        p_val = np.mean(interaction_diffs >= 0) * 2

    p_val = min(p_val, 1.0)
    ci_lower = np.percentile(interaction_diffs, 2.5)
    ci_upper = np.percentile(interaction_diffs, 97.5)

    return mean_interaction, ci_lower, ci_upper, p_val

def rdm_analysis_anchor(pca_data_dict: dict[str, PCAData], **kwargs):
    pca_data_dict_filtered = {}
    for name, pca_data in pca_data_dict.items():
        if 'anchor' in name:
            pca_data_dict_filtered[name] = pca_data
    kwargs['rdm_type'] = 'anchor'
    rdm_matrices_full_path = kwargs.get('output_dir') / 'rdm_cross_full' / 'full-rdm_matrices.pkl'
    kwargs['output_dir'] = kwargs.get('output_dir') / 'rdm_cross_anchor'
    matrices_anchor = _rdm_analysis(pca_data_dict_filtered, **kwargs)
    if rdm_matrices_full_path.exists():
        with open(rdm_matrices_full_path, 'rb') as f:
            matrices_full = pkl.load(f)
        ref_full = matrices_full["TwoPhoton_full"]
        m1_full = matrices_full["RetinodivnormGaborNet_full"]
        m2_full = matrices_full["GaborStimulus_full"]

        ref_anchor = matrices_anchor["TwoPhoton_anchors"]
        m1_anchor = matrices_anchor["RetinodivnormGaborNet_anchors"]
        m2_anchor = matrices_anchor["GaborStimulus_anchors"]

        # 3. Calculate the interaction!
        print("\n--- Running Interaction Test ---")
        res = interaction_bootstrap_rdm_difference(
            ref_full, m1_full, m2_full,
            ref_anchor, m1_anchor, m2_anchor,
            n_boot=1000
        )

        if res:
            diff, ci_l, ci_u, p_val = res
            print(f"Interaction: RetinoGaborNet vs GaborStimulus (Full -> Anchor)")
            print(f"Mean ΔΔr: {diff:.4f} | 95% CI: [{ci_l:.4f}, {ci_u:.4f}] | p-value: {p_val:.4f}")


def rdm_analysis_full(pca_data_dict: dict[str, PCAData], **kwargs):
    pca_data_dict_filtered = {}
    for name, pca_data in pca_data_dict.items():
        if 'full' in name:
            pca_data_dict_filtered[name] = pca_data
    kwargs['rdm_type'] = 'full'
    kwargs['output_dir'] = kwargs.get('output_dir') / 'rdm_cross_full'
    return _rdm_analysis(pca_data_dict_filtered, **kwargs)


def rdm_analysis_ori(pca_data_dict: dict[str, PCAData], **kwargs):
    kwargs['rdm_type'] = 'full'
    kwargs['output_dir'] = kwargs.get('output_dir') / 'rdm_cross_full'
    kwargs['ori'] = True
    print('Starting rdm_analysis_ori...')
    return _rdm_analysis(pca_data_dict, **kwargs)


def _rdm_analysis(pca_data_dict: dict[str, PCAData], **kwargs):
    rdms = []
    rdm_vectors = {}
    rdm_matrices = {}

    output_dir = kwargs.get('output_dir')
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, pca_data in pca_data_dict.items():
        data_source = pca_data.data_source
        sorted_arr = pca_data.pca_data.index.drop_duplicates().to_numpy()

        numpy_data = pca_data.raw_data.to_numpy()
        names = pca_data.raw_data.index.to_numpy()

        session_labels, min_trials = make_session_labels(names)

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

        if kwargs.get('ori', False) is False:
            unsorted_labels = rdm_cv.pattern_descriptors['conds']
            desired = sorted_arr
            name_to_idx = {name: i for i, name in enumerate(unsorted_labels)}
            reorder_indices = np.array([name_to_idx[name] for name in desired])
            rdm_cv.reorder(reorder_indices)

        rdm_cv.rdm_descriptors['name'] = [data_source]
        rdms.append(rdm_cv)

        rdm_vectors[key] = rdm_cv.get_vectors()[0]
        rdm_matrices[key] = rdm_cv.get_matrices()[0]

        fig, ax, _ = rdv.show_rdm(
            rdm_cv,
            show_colorbar='panel',
            pattern_descriptor='conds',
            rdm_descriptor=data_source,
            figsize=(10, 10),
            cmap='rocket'
        )
        plt.title(f"{key}", pad=20)
        plt.savefig(output_dir / f"{key}.svg", bbox_inches='tight')
        plt.close(fig)

    names = list(rdm_vectors.keys())
    n = len(names)
    comparison_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    # Compute correlation and p-values
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            r, p = spearmanr(rdm_vectors[n1], rdm_vectors[n2])
            comparison_matrix[i, j] = r
            p_matrix[i, j] = p

    # Bootstrap routine
    print(f"\n=================== Bootstrap Analysis: {kwargs.get('rdm_type', 'Default')} ===================")
    if kwargs.get('ori', False) is False:
        ref_name = next((name for name in names if 'TwoPhoton' in name), None)
    else:
        ref_name = next((name for name in names if 'Neural-State Space' in name), None)

    bootstrap_results = {}
    if ref_name:
        models = [m for m in names if m != ref_name]
        raw_results = {}  # lbl -> (diff, ci_l, ci_u, raw_p)

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                res = condition_bootstrap_rdm_difference(
                    rdm_matrices[ref_name],
                    rdm_matrices[m1],
                    rdm_matrices[m2],
                    n_boot=1000
                )
                if res is not None:
                    diff, ci_l, ci_u, p_val = res
                    m1_lbl = m1.replace('_full', '').replace('_anchor', '')
                    m2_lbl = m2.replace('_full', '').replace('_anchor', '')
                    lbl = f"{m1_lbl}\nvs\n{m2_lbl}"
                    raw_results[lbl] = (diff, ci_l, ci_u, p_val)
        if raw_results:
            labels = list(raw_results.keys())
            raw_pvals = [raw_results[lbl][3] for lbl in labels]
            reject, corrected_pvals, _, _ = multipletests(
                raw_pvals, alpha=0.05, method='fdr_bh'
            )

            for lbl, corrected_p, rej in zip(labels, corrected_pvals, reject):
                diff, ci_l, ci_u, raw_p = raw_results[lbl]
                bootstrap_results[lbl] = (diff, ci_l, ci_u, raw_p, corrected_p, rej)

                print(f"[{ref_name}] {lbl.replace(chr(10), ' ')}:")
                print(f"  -> Mean Diff: {diff:.4f} | 95% CI: [{ci_l:.4f}, {ci_u:.4f}]")
                print(f"  -> raw p = {raw_p:.4f} | FDR-corrected p = {corrected_p:.4f} "
                      f"| significant @ α=0.05: {rej}\n")

            pd.DataFrame({
                'comparison': [l.replace(chr(10), ' ') for l in labels],
                'mean_diff': [bootstrap_results[l][0] for l in labels],
                'ci_lower': [bootstrap_results[l][1] for l in labels],
                'ci_upper': [bootstrap_results[l][2] for l in labels],
                'raw_p': [bootstrap_results[l][3] for l in labels],
                'fdr_corrected_p': [bootstrap_results[l][4] for l in labels],
                'significant_fdr': [bootstrap_results[l][5] for l in labels],
            }).to_csv(output_dir / f'bootstrap_comparisons_{kwargs.get("rdm_type", "")}_fdr.csv', index=False)

        plot_bootstrap_comparisons(bootstrap_results, output_dir, rdm_type=kwargs.get("rdm_type", ""))
    else:
        print("No TwoPhoton baseline found to bootstrap against.")
    print("=========================================================================\n")

    _plot_stability(
        comparison_matrix,
        names,
        output_dir,
        name=f'Representational_Stability_{kwargs.get("rdm_type")}',
        show=True,
        full_data=True,
        p_matrix=p_matrix
    )
    with open(kwargs['output_dir'] / f'{kwargs['rdm_type']}-rdm_matrices.pkl', 'wb') as f:
        pkl.dump(rdm_matrices, f)
    return rdm_matrices


def rdm_analysis_subsets(pca_data_dict: dict[str, PCAData], **kwargs):
    output_dir = kwargs.get('output_dir') / 'rdm_cross_subsets'
    output_dir.mkdir(parents=True, exist_ok=True)

    cycles = defaultdict(dict)
    for key, pca_data in pca_data_dict.items():
        if 'subset' not in key: continue
        data_source = pca_data.data_source
        cycle_name = key[len(data_source) + 1:]
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

            unsorted_labels = rdm_cv.pattern_descriptors['conds']
            name_to_idx = {name: i for i, name in enumerate(unsorted_labels)}
            reorder_indices = np.array([
                name_to_idx[name] for name in sorted_arr
                if name in name_to_idx
            ])
            rdm_cv.reorder(reorder_indices)

            fig, ax, _ = rdv.show_rdm(
                rdm_cv,
                show_colorbar='panel',
                pattern_descriptor='conds',
                figsize=(10, 10),
                cmap='rocket'
            )
            plt.title(f"{data_source} - {cycle_name}", pad=20)
            plt.savefig(cycle_output_dir / f"{data_source}.svg", bbox_inches='tight')
            plt.close(fig)

            rdm_vectors[data_source] = rdm_cv.get_vectors()[0]

        source_names = list(rdm_vectors.keys())
        n = len(source_names)
        stability_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))

        for i, n1 in enumerate(source_names):
            for j, n2 in enumerate(source_names):
                r, p = spearmanr(rdm_vectors[n1], rdm_vectors[n2])
                stability_matrix[i, j] = r
                p_matrix[i, j] = p

        _plot_stability(
            stability_matrix,
            source_names,
            cycle_output_dir,
            name=f'Stability_{cycle_name}',
            show=False,
            full_data=False,
            p_matrix=p_matrix
        )

        all_stability_matrices.append(stability_matrix)
        stability_names = source_names

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


def make_session_labels(cond_labels: np.ndarray) -> np.ndarray:
    unique_conds, counts = np.unique(cond_labels, return_counts=True)
    min_trials = counts.min()
    session_labels = np.zeros(len(cond_labels), dtype=int)
    for cond in unique_conds:
        mask = cond_labels == cond
        n = mask.sum()
        indices = np.where(mask)[0]
        session_labels[indices] = np.arange(n) % min_trials
    return session_labels, min_trials


def _plot_stability(matrix, labels, output_dir, name, show, full_data, p_matrix=None):
    """
    Plots the stability correlation heatmap with FDR-corrected p-value
    annotations and exports raw + corrected p-value matrices to .csv.
    """
    plt.figure(figsize=(12, 10))
    save_name = name + "_full" if full_data else name

    df_corr = pd.DataFrame(matrix, index=labels, columns=labels)
    df_corr.to_csv(output_dir / f"{save_name}_correlations.csv")

    if p_matrix is not None:
        df_p_raw = pd.DataFrame(p_matrix, index=labels, columns=labels)
        df_p_raw.to_csv(output_dir / f"{save_name}_p_values_raw.csv")

        p_matrix_corrected, _ = fdr_correct_symmetric_matrix(p_matrix)
        df_p_corr = pd.DataFrame(p_matrix_corrected, index=labels, columns=labels)
        df_p_corr.to_csv(output_dir / f"{save_name}_p_values_fdr.csv")

        annot_labels = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                r = matrix[i, j]
                if i == j:
                    annot_labels[i, j] = f"{r:.2f}"
                else:
                    p_corr = p_matrix_corrected[i, j]
                    p_str = "p < .001" if p_corr < 0.001 else f"p = {p_corr:.3f}"
                    annot_labels[i, j] = f"{r:.2f}\n{p_str} (FDR)"

        sns.heatmap(matrix, annot=annot_labels, fmt="", cmap='viridis',
                    xticklabels=labels, yticklabels=labels, square=True,
                    cbar_kws={'label': "Spearman Correlation (r)"})
    else:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap='viridis',
                    xticklabels=labels, yticklabels=labels, square=True,
                    cbar_kws={'label': "Spearman Correlation (r)"})

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(f"{name.replace('_', ' ')}", pad=20, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_name}.svg", bbox_inches='tight')
    plt.savefig(output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()