import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from helper.PlotCreatorMixInHelper import PlotCreatorMixInHelper


class PlotCreatorMixIn(PlotCreatorMixInHelper):

    def _create_2d_plots(self, pca, labels, pca_type, plot_name):
        data, labels, _ = self.sort_morphs(pca, labels)
        numeric_index = np.arange(len(labels))

        plt.figure(figsize=(12.5, 7.5))
        loop_data = np.vstack([data, data[0]])
        plt.plot(loop_data[:, 0], loop_data[:, 1],
                 color='gray', alpha=0.4, linestyle='--', zorder=1, label='Morph Path')
        sc = plt.scatter(data[:, 0], data[:, 1],
                         c=numeric_index, cmap='viridis', s=60, alpha=0.8,
                         edgecolors='white', zorder=2)
        is_anchor = labels['stim_type'] == 'anchor'
        for i in np.where(is_anchor)[0]:
            name = labels['morph_name'].iloc[i]
            plt.text(data[i, 0], data[i, 1] + 0.5, name, fontsize=10,
                     fontweight='bold', ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))
        cbar = plt.colorbar(sc)
        cbar.set_ticks(numeric_index)
        cbar.set_ticklabels(labels['morph_name'])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'{plot_name} 2D {pca_type} (Closed Loop)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()
    #
    # def _create_3d_plots(self, pca, labels, pca_type, plot_name):
    #     unique_labels, numeric_label = np.unique(labels['morph_name'], return_inverse=True)
    #
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     sc = ax.scatter(
    #         pca[:, 0],
    #         pca[:, 1],
    #         pca[:, 2],
    #         c=numeric_label,
    #         cmap='hsv',
    #         s=30,
    #         alpha=0.6,  # Make background slightly transparent
    #         edgecolors='none'
    #     )
    #
    #     # 2. Overlay the "Masked" points with a distinct marker
    #     mask_array = np.array(name_mask)  # Ensure it's a numpy array for indexing
    #
    #     ax.scatter(
    #         pca[mask_array, 0],
    #         pca[mask_array, 1],
    #         pca[mask_array, 2],
    #         c='black',  # Solid color to make them pop
    #         marker='X',  # 'X' shape is very clear
    #         s=100,  # Larger size
    #         label='Full Morphs (00/21)',
    #         depthshade=False  # Keeps the color vivid in 3D
    #     )
    #
    #     for i, name in enumerate(self.labels):
    #         if self.name_mask[i]:
    #             # ax.text(x, y, z, string)
    #             ax.text(
    #                 self.pca[i, 0],
    #                 self.pca[i, 1],
    #                 self.pca[i, 2],
    #                 name,  # The filename
    #                 fontsize=8,
    #                 fontweight='bold',
    #                 color='black'
    #             )
    #
    #     fig.colorbar(sc, label='degrees (°)')
    #     ax.set_xlabel('PC1')
    #     ax.set_ylabel('PC2')
    #     ax.set_zlabel('PC3')
    #     ax.set_title(f'{self.plot_name} 3D {self.pca_type}')
    #     ax.legend()  # Show the label for the 'X' markers
    #     plt.show()

    @staticmethod
    def _create_interactive_plot(pca, labels, pca_type, plot_name, output_dir):
        # 1. Setup Colors with a Gradient for matches
        for pair_key in np.unique(labels['pair_key'].dropna()):
            search_term = pair_key

            # We find all matching indices first
            matches = [i for i, label in enumerate(labels['pair_key']) if search_term in str(label)]
            num_matches = len(matches)

            # Create the color array (initialized to 0 for Gray)
            plot_colors = np.zeros(len(labels))

            if num_matches > 0:
                # Assign values from 0.1 to 1.0 to the matches
                # This allows the 'RdYlGn' scale to actually show a gradient
                plot_colors[matches] = np.linspace(0.1, 1.0, num_matches)

            # 2. Filter "Full Morphs" (Your existing logic)
            mask_array = labels['stim_type'] != 'morph'
            mask_indices = np.where(mask_array)[0]
            seen_coords = set()
            final_mask_x, final_mask_y, final_mask_z, final_mask_text = [], [], [], []

            for idx in mask_indices:
                coord = tuple(np.round(pca[idx], 2))
                if coord not in seen_coords:
                    final_mask_x.append(pca[idx, 0])
                    final_mask_y.append(pca[idx, 1])
                    final_mask_z.append(pca[idx, 2])
                    final_mask_text.append(labels['morph_name'][idx])
                    seen_coords.add(coord)

            # 3. Create the Figure
            fig = go.Figure()

            # Trace 1: All points
            fig.add_trace(go.Scatter3d(
                x=pca[:, 0], y=pca[:, 1], z=pca[:, 2],
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
                text=labels,
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
                title=f"Highlighting: {search_term} (Gray=Other, Red->Green=Matches){pca_type}",
                scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                margin=dict(l=0, r=0, b=0, t=50)
            )

            filename = f"{output_dir}/{plot_name}_{pair_key}.html"
            fig.write_html(filename)
            print(f"Plot saved to {filename}")

    @staticmethod
    def calculate_distances(pca, labels, pca_type):
        name_stems = labels['pair_key'].dropna().values
        all_distances = []
        all_distances_cum = []
        for stem in np.unique(name_stems):
            mask = labels['pair_key'] == stem
            anchor_mask = labels['morph_name'].isin(stem.split('__'))
            mask = mask | anchor_mask
            pca_masked = pca[mask, :2]
            distances = [np.linalg.norm(pca_masked[i + 1] - pca_masked[i]) for i in range(len(pca_masked) - 1)]
            cumsum_differences = np.cumsum(distances)
            scaled_distances_cum = [dis / cumsum_differences[-1] for dis in cumsum_differences]
            scaled_distances = [dis / cumsum_differences[-1] for dis in distances]
            # plt.plot(scaled_distances_cum)
            # plt.title(f'{label} distances between points (cumulative) {pca_type}')
            # plt.show()
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
        plt.title(f"Mean Distances scaled {pca_type}")
        plt.ylim(0, 1)
        plt.legend()
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
        plt.title(f"Mean Distance (cumulative) {pca_type}")
        plt.show()
