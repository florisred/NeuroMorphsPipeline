from pathlib import Path, PosixPath
import numpy as np
import plotly.graph_objects as go
from data_objects.pca_data import PCAData
from utils.utils import find_max_separation


def create_interactive_plot(pca_data_dict: dict[str, PCAData], **kwargs):
    """
    pc_indices: tuple of 3 integers representing the PCs to plot (e.g., (0, 2, 3))
    """
    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path):
        raise ValueError('output_dir not provided')

    output_dir = output_dir / 'interactive'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract indices for readability
    components = find_max_separation(pca_data_dict=pca_data_dict, num_comps=3, filter='full')

    for k, pca_data in pca_data_dict.items():
        if 'full' not in k: continue
        data = pca_data.pca_data
        metadata = pca_data.metadata
        ix, iy, iz = components[k]

        for search_term in metadata.get_pair_keys():
            matches, num_matches = metadata.find_matching_pair_keys(search_term)
            plot_colors = np.zeros(len(data))

            if num_matches > 0:
                plot_colors[matches] = np.linspace(0.1, 1.0, num_matches)

            mask_array = metadata.anchor_mask
            mask_indices = np.where(mask_array)[0]
            seen_coords = set()
            final_mask_x, final_mask_y, final_mask_z, final_mask_text = [], [], [], []

            for idx in mask_indices:
                # Select specific PCs for coordinate rounding/uniqueness check
                coord = tuple(np.round(data.iloc[idx, [ix, iy, iz]], 2))
                if coord not in seen_coords:
                    final_mask_x.append(data.iloc[idx, ix])
                    final_mask_y.append(data.iloc[idx, iy])
                    final_mask_z.append(data.iloc[idx, iz])
                    final_mask_text.append(metadata.morph_names.iloc[idx])
                    seen_coords.add(coord)

            fig = go.Figure()

            # Trace 1: All points using selected PC indices
            fig.add_trace(go.Scatter3d(
                x=data.iloc[:, ix], y=data.iloc[:, iy], z=data.iloc[:, iz],
                mode='markers',
                marker=dict(
                    size=6,
                    color=plot_colors,
                    colorscale=[
                        [0, 'rgb(200, 200, 200)'],
                        [0.001, 'rgb(200, 200, 200)'],
                        [0.1, 'red'],
                        [0.5, 'yellow'],
                        [1.0, 'green']
                    ],
                    showscale=True,
                    colorbar=dict(title="Gradient")
                ),
                text=metadata.morph_names,
                name='All Images',
                hoverinfo='text'
            ))

            # Trace 2: Black 'x' markers
            fig.add_trace(go.Scatter3d(
                x=final_mask_x, y=final_mask_y, z=final_mask_z,
                mode='markers',
                marker=dict(size=7, color='black', symbol='x', line=dict(width=1, color='white')),
                text=final_mask_text,
                name='Unique Full Morphs',
                hoverinfo='text'
            ))

            # Layout: Update axis titles dynamically
            fig.update_layout(
                template="plotly_white",
                title=f"Highlighting: {search_term} (PCs: {ix + 1}, {iy + 1}, {iz + 1})",
                scene=dict(
                    xaxis_title=f'PC{ix + 1}',
                    yaxis_title=f'PC{iy + 1}',
                    zaxis_title=f'PC{iz + 1}'
                ),
                margin=dict(l=0, r=0, b=0, t=50)
            )

            filename = output_dir / f'{pca_data.name}_{search_term}_PC{ix + 1}{iy + 1}{iz + 1}.html'
            fig.write_html(filename)