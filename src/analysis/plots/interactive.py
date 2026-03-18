from pathlib import Path, PosixPath
import numpy as np
import plotly.graph_objects as go
from data_objects.pca_data import PCAData

def create_interactive_plot(pca_data_dict: dict[str, PCAData], **kwargs):
    output_dir = kwargs.get('output_dir')
    if not output_dir or not isinstance(output_dir, Path): raise ValueError('output_dir not provided')
    output_dir = output_dir / 'interactive'
    output_dir.mkdir(parents=True, exist_ok=True)

    for k, pca_data in pca_data_dict.items():
        if 'full' not in k: continue
        data = pca_data.pca_data
        metadata = pca_data.metadata
        for search_term in metadata.get_pair_keys():
            matches, num_matches = metadata.find_matching_pair_keys(search_term)
            plot_colors = np.zeros(len(data))

            if num_matches > 0: plot_colors[matches] = np.linspace(0.1, 1.0, num_matches)

            mask_array = metadata.get_anchor_mask()  # self.labels['stim_type'] != 'morph'
            mask_indices = np.where(mask_array)[0]
            seen_coords = set()
            final_mask_x, final_mask_y, final_mask_z, final_mask_text = [], [], [], []

            for idx in mask_indices:
                coord = tuple(np.round(data.iloc[idx], 2))
                if coord not in seen_coords:
                    final_mask_x.append(data.iloc[idx, 0])
                    final_mask_y.append(data.iloc[idx, 1])
                    final_mask_z.append(data.iloc[idx, 2])
                    final_mask_text.append(metadata.get_morph_names().iloc[idx])
                    seen_coords.add(coord)

            # 3. Create the Figure
            fig = go.Figure()

            # Trace 1: All points
            fig.add_trace(go.Scatter3d(
                x=data.iloc[:, 0], y=data.iloc[:, 1], z=data.iloc[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=plot_colors,
                    # CUSTOM COLORSCALE:
                    # 0.0 is Gray
                    # 0.1 to 1.0 is the Red-Yellow-Green gradient
                    colorscale=[
                        [0, 'rgb(200, 200, 200)'],  # 0 is Gray
                        [0.001, 'rgb(200, 200, 200)'],  # Buffer to keep gray solid
                        [0.1, 'red'],  # Start of gradient
                        [0.5, 'yellow'],  # Middle
                        [1.0, 'green']  # End
                    ],
                    showscale=True,
                    colorbar=dict(title="Gradient")
                ),
                text=metadata.get_morph_names(),
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

            # 4. Layout
            fig.update_layout(
                template="plotly_white",
                title=f"Highlighting: {search_term} from {pca_data.name}",
                scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                margin=dict(l=0, r=0, b=0, t=50)
            )

            filename = output_dir / f'{pca_data.name}_{search_term}.html'
            fig.write_html(filename)
            print(f"Plot saved to {filename}")