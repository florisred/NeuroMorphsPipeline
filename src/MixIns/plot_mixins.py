from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from data_objects.pca_data import PCAData


class Plot2DMixIn:
    @staticmethod
    def create_2d_plots(pca_data: PCAData, output_dir: Path):
        output_dir = output_dir / '2D'
        output_dir.mkdir(parents=True, exist_ok=True)
        data = pca_data.pca_data
        metadata = pca_data.metadata
        numeric_index = pca_data.get_numeric_index()

        plt.figure(figsize=(12.5, 7.5))
        loop_data = np.vstack([data, data.iloc[0]])
        plt.plot(loop_data[:, 0], loop_data[:, 1],
                 color='gray', alpha=0.4, linestyle='--', zorder=1, label='Morph Path')
        sc = plt.scatter(data.iloc[:, 0], data.iloc[:, 1],
                         c=numeric_index, cmap='viridis', s=60, alpha=0.8,
                         edgecolors='white', zorder=2)
        is_anchor = metadata.get_anchor_mask()
        for i in np.where(is_anchor)[0]:
            name = metadata.get_morph_names().iloc[i]
            plt.text(data.iloc[i, 0], data.iloc[i, 1] + 0.5, name, fontsize=10,
                     fontweight='bold', ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))
        cbar = plt.colorbar(sc)
        cbar.set_ticks(numeric_index)
        cbar.set_ticklabels(metadata.get_morph_names())
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'{pca_data.name} 2D')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(output_dir / f'{pca_data.name}.png')
        plt.show()

class PlotInteractiveMixIn:

    @staticmethod
    def create_interactive_plot(pca_data: PCAData, output_dir: Path):
        output_dir = output_dir / 'interactive'
        output_dir.mkdir(parents=True, exist_ok=True)
        data = pca_data.pca_data
        metadata = pca_data.metadata
        for search_term in metadata.get_pair_keys():
            matches, num_matches = metadata.find_matching_pair_keys(search_term)
            plot_colors = np.zeros(len(data))

            if num_matches > 0: plot_colors[matches] = np.linspace(0.1, 1.0, num_matches)

            mask_array = metadata.get_anchor_mask()#self.labels['stim_type'] != 'morph'
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


class PlotDistancesMixIn:
    @staticmethod
    def calculate_distances(pca_data: PCAData, output_dir: Path):
        output_dir = output_dir / 'distances'
        output_dir.mkdir(parents=True, exist_ok=True)
        data = pca_data.pca_data
        metadata = pca_data.metadata
        name_stems = metadata.get_pair_keys(unique=True, values=True, dropna=True)
        all_distances = []
        all_distances_cum = []
        pair_keys = metadata.get_pair_keys(unique=False, values=False, dropna=False)

        for stem in name_stems:
            mask = pair_keys == stem
            anchor_mask = metadata.get_morph_names().isin(stem.split('__'))
            mask = mask | anchor_mask
            pca_masked = data[mask]
            distances = [np.linalg.norm(pca_masked.iloc[i+1] - pca_masked.iloc[i]) for i in range(len(pca_masked)-1)]
            distances_with_0 = [0.0]
            for distance in distances: distances_with_0.append(distance)
            cumsum_differences = np.cumsum(distances_with_0)
            scaled_distances_cum = [dis / cumsum_differences[-1] for dis in cumsum_differences]
            scaled_distances = [dis / cumsum_differences[-1] for dis in distances]
            all_distances.append(scaled_distances)
            all_distances_cum.append(scaled_distances_cum)

        mean_distance = np.mean(all_distances, axis=0)
        sem_distance = np.std(all_distances, axis=0)
        mean_distance_cum = np.mean(all_distances_cum, axis=0)
        sem_distance_cum = np.std(all_distances_cum, axis=0)

        # Plot 1: Mean Distances
        plt.figure(figsize=(10, 7.5))
        plt.plot(mean_distance, label='Mean Distance', color='blue')
        # Adding the SEM/STD ribbon
        plt.fill_between(
            range(len(mean_distance)),
            mean_distance - sem_distance,
            mean_distance + sem_distance,
            color='blue', alpha=0.2, label='± 1 STD'
        )
        title = f"Mean Distances scaled {pca_data.name}"
        plt.ylim(0, 1)
        plt.legend()
        plt.title(title)
        plt.savefig(output_dir / f'{title}.png')
        plt.show()

        # Plot 2: Cumulative Mean Distances
        plt.figure(figsize=(10, 7.5))
        plt.plot(mean_distance_cum, label='Mean Cumulative', color='green')
        # Adding the SEM/STD ribbon
        plt.fill_between(
            range(len(mean_distance_cum)),
            mean_distance_cum - sem_distance_cum,
            mean_distance_cum + sem_distance_cum,
            color='green', alpha=0.2
        )
        title = f"Mean Distance (cumulative) {pca_data.name}"
        plt.title(title)
        plt.savefig(output_dir / f'{title}.png')
        plt.show()