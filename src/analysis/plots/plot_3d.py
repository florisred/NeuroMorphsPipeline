import matplotlib.pyplot as plt
import colorsys
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from data_objects.pca_data import PCAData

HIGHLIGHT_COLOR = "#E63946"
FADED_ANCHOR_COLOR = "#aaaaaa"

# Saturation and lightness controls for the per-pair pastel palette.
# Hues are spread evenly around the colour wheel; S and L are fixed to keep
# every colour equally muted so none pops out over the highlighted pair.
_FADED_SATURATION = 0.35   # 0 = grey, 1 = fully saturated
_FADED_LIGHTNESS  = 0.72   # higher = lighter / more washed-out
_FADED_ALPHA      = 0.35   # point opacity


def _get_single_row(df: pd.DataFrame, name: str) -> pd.Series | None:
    """Safely retrieve a single row by index name, handling duplicate indices."""
    if name not in df.index:
        return None
    result = df.loc[name]
    return result.iloc[0] if isinstance(result, pd.DataFrame) else result


def _anchor_distance(pca_data: PCAData, pair_key: str) -> float:
    """Euclidean distance between the two anchors of a pair in PC space."""
    parts = pair_key.split("__")
    if len(parts) != 2:
        print(f"Warning: unexpected pair_key format '{pair_key}'")
        return 0.0
    src, dst = parts
    data = pca_data.get_data_components(n_components=3)
    src_row = _get_single_row(data, src)
    dst_row = _get_single_row(data, dst)
    if src_row is None or dst_row is None:
        print(f"Warning: anchor not found — src='{src}' found={src_row is not None}, dst='{dst}' found={dst_row is not None}")
        return 0.0
    return float(np.linalg.norm(src_row.values - dst_row.values))


def _find_best_pair(pca_data_dict: dict[str, PCAData]) -> str:
    """Find the pair_key with the highest mean anchor distance across all data sources."""
    all_pair_key_sets = [
        set(pca_data.metadata.get_pair_keys(unique=True, dropna=True, values=True))
        for pca_data in pca_data_dict.values()
    ]
    shared_pairs = set.intersection(*all_pair_key_sets)

    mean_distances = {
        pk: np.mean([
            _anchor_distance(pca_data, pk)
            for pca_data in pca_data_dict.values()
        ])
        for pk in shared_pairs
    }

    best = max(mean_distances, key=mean_distances.get)
    print(f"Pair distances: { {k: round(v, 2) for k, v in sorted(mean_distances.items(), key=lambda x: -x[1])} }")
    print(f"Selected pair: '{best}' (mean distance={mean_distances[best]:.2f})")
    return best


def _parse_src_ratio(morph_name: str, src_anchor: str) -> float:
    """
    Extract the src anchor's blend ratio from a morph name.
    E.g. 'bark_0.86_brick_0.14' with src_anchor='bark' → 0.86.
    Returns 0.5 as fallback if parsing fails.
    """
    parts = morph_name.split("_")
    try:
        src_idx = parts.index(src_anchor)
        return float(parts[src_idx + 1])
    except (ValueError, IndexError):
        return 0.5


def _build_pair_color_map(all_pair_keys: list[str]) -> dict[str, str]:
    """
    Assign a unique muted HLS colour to every pair_key.
    Hues are spread evenly around the wheel; saturation and lightness are
    fixed so every group looks equally faded next to the red highlight.
    """
    n = len(all_pair_keys)
    color_map = {}
    for i, pk in enumerate(sorted(all_pair_keys)):   # sort for determinism
        hue = i / n
        rgb = colorsys.hls_to_rgb(hue, _FADED_LIGHTNESS, _FADED_SATURATION)
        color_map[pk] = mcolors.to_hex(rgb)
    return color_map


def create_3d_plot(pca_data_dict: dict[str, PCAData], **kwargs):
    output_dir = kwargs.get("output_dir") / '3d_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    with_subsets = kwargs.get('make_3d_plots_with_subsets', True)

    for key, pca_data in pca_data_dict.items():
        # Build a stable colour map over ALL pair keys for this data source
        if not with_subsets:
            if 'subset' in key: continue
        all_pair_keys = pca_data.metadata.get_pair_keys(unique=True, dropna=True, values=True)
        pair_color_map = _build_pair_color_map(list(all_pair_keys))

        for pair_key in pca_data.metadata.get_pair_keys():
            src_anchor, dst_anchor = pair_key.split("__")
            data = pca_data.get_data_components(n_components=3)
            metadata_df = pca_data.metadata_df
            anchor_mask = pca_data.metadata.anchor_mask.values

            row_pair_keys = pca_data.metadata.get_pair_keys(unique=False, dropna=False, values=True)
            is_highlight_morph = np.array([pk == pair_key for pk in row_pair_keys]) & ~anchor_mask
            is_highlight_anchor = np.array([name in (src_anchor, dst_anchor) for name in data.index]) & anchor_mask

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # --- faded background morphs: one colour per other pair_key ---
            other_pair_keys = [pk for pk in pair_color_map if pk != pair_key]
            legend_other_handles = []

            for other_pk in other_pair_keys:
                other_color = pair_color_map[other_pk]
                other_morph_mask = (
                    np.array([pk == other_pk for pk in row_pair_keys]) & ~anchor_mask
                )
                other_idx = np.where(other_morph_mask)[0]
                if len(other_idx) == 0:
                    continue

                ax.scatter(
                    data.iloc[other_idx, 0],
                    data.iloc[other_idx, 1],
                    data.iloc[other_idx, 2],
                    c=other_color, alpha=_FADED_ALPHA, s=15, zorder=1,
                )

                # One legend entry per other pair (label = "A ↔ B")
                src_o, dst_o = other_pk.split("__")
                legend_other_handles.append(
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=other_color,
                           markersize=6, alpha=0.8, label=f"{src_o} ↔ {dst_o}")
                )

            # --- faded background anchors ---
            faded_anchor_idx = np.where(~is_highlight_anchor & anchor_mask)[0]
            ax.scatter(
                data.iloc[faded_anchor_idx, 0],
                data.iloc[faded_anchor_idx, 1],
                data.iloc[faded_anchor_idx, 2],
                c=FADED_ANCHOR_COLOR, alpha=0.3, s=80, zorder=2, marker="*",
            )

            # --- highlighted morph points ---
            highlight_morph_idx = np.where(is_highlight_morph)[0]
            ax.scatter(
                data.iloc[highlight_morph_idx, 0],
                data.iloc[highlight_morph_idx, 1],
                data.iloc[highlight_morph_idx, 2],
                c=HIGHLIGHT_COLOR, alpha=0.8, s=40, zorder=4,
            )

            # --- sequential lines: src_anchor → morph_1 → morph_2 → ... → dst_anchor ---
            sorted_morph_idx = sorted(
                highlight_morph_idx,
                key=lambda i: _parse_src_ratio(data.index[i], src_anchor),
                reverse=True,
            )

            src_row = _get_single_row(data, src_anchor)
            dst_row = _get_single_row(data, dst_anchor)

            chain = []
            if src_row is not None:
                chain.append(src_row.values)
            for i in sorted_morph_idx:
                chain.append(data.iloc[i].values)
            if dst_row is not None:
                chain.append(dst_row.values)

            for a, b in zip(chain, chain[1:]):
                ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    [a[2], b[2]],
                    color=HIGHLIGHT_COLOR, alpha=0.25, linewidth=0.8, zorder=3,
                )

            # --- highlighted anchor points + labels ---
            for anchor_name in (src_anchor, dst_anchor):
                anchor_row = _get_single_row(data, anchor_name)
                if anchor_row is None:
                    continue
                ax.scatter(
                    float(anchor_row.iloc[0]),
                    float(anchor_row.iloc[1]),
                    float(anchor_row.iloc[2]),
                    c=HIGHLIGHT_COLOR, s=220, edgecolors="white", linewidths=1.4,
                    zorder=6, marker="*",
                )
                ax.text(
                    float(anchor_row.iloc[0]) + 0.5,
                    float(anchor_row.iloc[1]) + 0.5,
                    float(anchor_row.iloc[2]) + 0.5,
                    anchor_name,
                    fontsize=9, fontweight="bold", color=HIGHLIGHT_COLOR,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                )

            # --- legend ---
            primary_handles = [
                Line2D([0], [0], marker="*", color="w", markerfacecolor=HIGHLIGHT_COLOR,
                       markersize=12, label=f"{src_anchor} ↔ {dst_anchor} (highlighted)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor=HIGHLIGHT_COLOR,
                       markersize=7, label="Morph (highlighted)"),
                Line2D([0], [0], marker="*", color="w", markerfacecolor=FADED_ANCHOR_COLOR,
                       markersize=9, label="Other Full Textures"),
            ]
            ax.legend(
                handles=primary_handles + legend_other_handles,
                loc="upper left",
                fontsize=7,
                framealpha=0.85,
                title=f"Highlighted: {src_anchor} ↔ {dst_anchor}",
                title_fontsize=8,
            )

            ax.set_xlabel("PC 1", labelpad=8)
            ax.set_ylabel("PC 2", labelpad=8)
            ax.set_zlabel("PC 3", labelpad=8)
            ax.set_title(f"PCA – {key}", pad=12)
            ax.grid(True, linewidth=0.3, alpha=0.4)

            plt.tight_layout()
            new_output_dir = output_dir / pair_key
            new_output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(new_output_dir / f'3d_{key}.svg')
            plt.close()