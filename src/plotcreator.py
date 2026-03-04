import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


class PlotCreator:
    def __init__(self, plot_settings):
        self.plot_name = plot_settings["plot_name"]
        self.do_2d_plots = plot_settings["do_2d_plots"]
        self.do_3d_plots = plot_settings["do_3d_plots"]
        self.do_distances = plot_settings["do_distances"]
        self.do_interactive_plots = plot_settings["do_interactive_plots"]
        self.pca = []
        self.labels = pd.DataFrame()
        self.name_mask = []
        self.pca_type = None
        self.output_dir = plot_settings["data_dir"]



    def create_plots(self, pca, processed_names, pca_type, name_mask=None):


        #pca = pca.to_numpy()
        self.pca = pca
        self.labels = processed_names #pd.DataFrame(processed_names)
        if name_mask is None:
            name_mask = np.zeros(len(processed_names), dtype=bool)
        self.name_mask = name_mask
        self.pca_type = pca_type





        print("Creating plots...")
        if self.do_distances and self.pca_type in ['two_photon', 'pixel', 'gabor', 'neuropixel']:
            self.calculate_distances_half()
        if self.do_2d_plots and self.pca_type not in ['two_photon', 'pixel', 'gabor', 'neuropixel']:
            self._create_2d_plots()
        if self.do_3d_plots and self.pca_type in ['two_photon', 'pixel', 'gabor', 'neuropixel']:
            self._create_3d_plots()
        if self.do_interactive_plots and self.pca_type in ['two_photon', 'pixel', 'gabor', 'neuropixel']:
            self._create_interactive_plot()

    def _create_2d_plots(self):

        is_anchor = self.labels['stim_type'].isin(['anchor'])
        plt.figure(figsize=(10, 7))
        unique_labels, numeric_label = np.unique(self.labels['full_name'], return_inverse=True)
        # 1. Plot all points
        sc = plt.scatter(self.pca[:, 0], self.pca[:, 1],
                         c=numeric_label, cmap='viridis', s=50, alpha=0.8)

        # 2. Plot and Label the X markers
        # We iterate through the indices where is_anchor is True
        for i in np.where(is_anchor)[0]:
            name = self.labels['full_name'].iloc[i]
            x, y = self.pca[i, 0], self.pca[i, 1]

            # Draw the X
            #plt.scatter(x, y, marker='x', s=100, color='black', linewidths=2)

            # Add the text label slightly offset from the point
            plt.text(x - 1, y - 2, name, fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # 3. Handle the Colorbar with Name Ticks
        cbar = plt.colorbar(sc)
        cbar.set_ticks(range(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)
        cbar.set_label('Stimuli')

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'{self.plot_name} 2D {self.pca_type}')
        plt.grid(True)
        plt.show()


    def _create_3d_plots(self):
        unique_labels, numeric_label = np.unique(self.labels['full_name'], return_inverse=True)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # 1. Plot ALL points as small dots (background)
        sc = ax.scatter(
            self.pca[:, 0],
            self.pca[:, 1],
            self.pca[:, 2],
            c=numeric_label,
            cmap='hsv',
            s=30,
            alpha=0.6,  # Make background slightly transparent
            edgecolors='none'
        )

        # 2. Overlay the "Masked" points with a distinct marker
        mask_array = np.array(self.name_mask)  # Ensure it's a numpy array for indexing

        ax.scatter(
            self.pca[mask_array, 0],
            self.pca[mask_array, 1],
            self.pca[mask_array, 2],
            c='black',  # Solid color to make them pop
            marker='X',  # 'X' shape is very clear
            s=100,  # Larger size
            label='Full Morphs (00/21)',
            depthshade=False  # Keeps the color vivid in 3D
        )

        for i, name in enumerate(self.labels):
            if self.name_mask[i]:
                # ax.text(x, y, z, string)
                ax.text(
                    self.pca[i, 0],
                    self.pca[i, 1],
                    self.pca[i, 2],
                    name,  # The filename
                    fontsize=8,
                    fontweight='bold',
                    color='black'
                )

        fig.colorbar(sc, label='degrees (°)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'{self.plot_name} 3D {self.pca_type}')
        ax.legend()  # Show the label for the 'X' markers
        plt.show()



    def _create_interactive_plot(self):
        # 1. Setup Colors with a Gradient for matches
        for pair_key in self.labels['pair_key']:
            search_term = pair_key

            # We find all matching indices first
            matches = [i for i, label in enumerate(self.labels['pair_key']) if search_term in str(label)]
            num_matches = len(matches)

            # Create the color array (initialized to 0 for Gray)
            plot_colors = np.zeros(len(self.labels))

            if num_matches > 0:
                # Assign values from 0.1 to 1.0 to the matches
                # This allows the 'RdYlGn' scale to actually show a gradient
                plot_colors[matches] = np.linspace(0.1, 1.0, num_matches)

            # 2. Filter "Full Morphs" (Your existing logic)
            mask_array = self.labels['stim_type'] != 'morph'
            mask_indices = np.where(mask_array)[0]
            seen_coords = set()
            final_mask_x, final_mask_y, final_mask_z, final_mask_text = [], [], [], []

            for idx in mask_indices:
                coord = tuple(np.round(self.pca[idx], 2))
                if coord not in seen_coords:
                    final_mask_x.append(self.pca[idx, 0])
                    final_mask_y.append(self.pca[idx, 1])
                    final_mask_z.append(self.pca[idx, 2])
                    final_mask_text.append(self.labels[idx])
                    seen_coords.add(coord)

            # 3. Create the Figure
            fig = go.Figure()

            # Trace 1: All points
            fig.add_trace(go.Scatter3d(
                x=self.pca[:, 0], y=self.pca[:, 1], z=self.pca[:, 2],
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
                text=self.labels,
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
                title=f"Highlighting: {search_term} (Gray=Other, Red->Green=Matches){self.pca_type}",
                scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                margin=dict(l=0, r=0, b=0, t=50)
            )

            filename = f"{self.output_dir}/{self.plot_name}_{pair_key}.html"
            fig.write_html(filename)
            print(f"Plot saved to {filename}")



    def calculate_distances(self):
        name_stems = self.labels['pair_key'].values
        all_distances = []
        all_distances_cum = []
        for stem in np.unique(name_stems):
            mask = self.labels['pair_key'] == stem
            pca_masked = self.pca[mask, :2]
            distances = [np.linalg.norm(pca_masked[i+1] - pca_masked[i]) for i in range(len(pca_masked)-1)]
            cumsum_differences = np.cumsum(distances)
            scaled_distances_cum = [dis / cumsum_differences[-1] for dis in cumsum_differences]
            scaled_distances = [dis / cumsum_differences[-1] for dis in distances]
            # plt.plot(scaled_distances_cum)
            # plt.title(f'{label} distances between points (cumulative) {self.pca_type}')
            # plt.show()
            all_distances.append(scaled_distances)
            all_distances_cum.append(scaled_distances_cum)

        mean_distance = np.mean(all_distances, axis=0)
        sem_distance = np.std(all_distances, axis=0)
        mean_distance_cum = np.mean(all_distances_cum, axis=0)
        sem_distance_cum = np.std(all_distances_cum, axis=0)

        # Plot 1: Mean Distances
        plt.figure(figsize=(10, 5))
        plt.plot(mean_distance, label='Mean Distance', color='blue')
        # Adding the SEM/STD ribbon
        plt.fill_between(
            range(len(mean_distance)),
            mean_distance - sem_distance,
            mean_distance + sem_distance,
            color='blue', alpha=0.2, label='± 1 STD'
        )
        plt.title(f"Mean Distances scaled {self.pca_type}")
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

        # Plot 2: Cumulative Mean Distances
        plt.figure(figsize=(10, 5))
        plt.plot(mean_distance_cum, label='Mean Cumulative', color='green')
        # Adding the SEM/STD ribbon
        plt.fill_between(
            range(len(mean_distance_cum)),
            mean_distance_cum - sem_distance_cum,
            mean_distance_cum + sem_distance_cum,
            color='green', alpha=0.2
        )
        plt.title(f"Mean Distance (cumulative) {self.pca_type}")
        plt.show()
        tes = 1



    def calculate_distances_half(self):
        name_stems = self.labels['pair_key'].values
        all_distances = []
        all_distances_cum = []
        for stem in np.unique(name_stems):
            mask = self.labels['pair_key'] == stem
            pca_masked = self.pca[mask, :2]
            pca_masked= pca_masked[:5,:]
            distances = [np.linalg.norm(pca_masked[i+1] - pca_masked[i]) for i in range(len(pca_masked)-1)]
            cumsum_differences = np.cumsum(distances)
            scaled_distances_cum = [dis / cumsum_differences[-1] for dis in cumsum_differences]
            scaled_distances = [dis / cumsum_differences[-1] for dis in distances]
            plt.plot(scaled_distances_cum)
            plt.title(f'{stem.split('__')[0]} to 50/50 distance between points{self.pca_type}')
            plt.show()
            all_distances.append(scaled_distances)
            all_distances_cum.append(scaled_distances_cum)

        mean_distance = np.mean(all_distances, axis=0)
        sem_distance = np.std(all_distances, axis=0)
        mean_distance_cum = np.mean(all_distances_cum, axis=0)
        sem_distance_cum = np.std(all_distances_cum, axis=0)

        # Plot 1: Mean Distances
        plt.figure(figsize=(10, 5))
        plt.plot(mean_distance, label='Mean Distance', color='blue')
        # Adding the SEM/STD ribbon
        plt.fill_between(
            range(len(mean_distance)),
            mean_distance - sem_distance,
            mean_distance + sem_distance,
            color='blue', alpha=0.2, label='± 1 STD'
        )
        plt.title(f"Mean Distances scaled {self.pca_type}")
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

        # Plot 2: Cumulative Mean Distances
        plt.figure(figsize=(10, 5))
        plt.plot(mean_distance_cum, label='Mean Cumulative', color='green')
        # Adding the SEM/STD ribbon
        plt.fill_between(
            range(len(mean_distance_cum)),
            mean_distance_cum - sem_distance_cum,
            mean_distance_cum + sem_distance_cum,
            color='green', alpha=0.2
        )
        plt.title(f"Mean Distance (cumulative) {self.pca_type}")
        plt.show()