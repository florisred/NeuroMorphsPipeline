import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


class PlotCreator:
    def __init__(self, plot_settings):
        self.plot_name = plot_settings["plot_name"]
        self.do_2d_plots = plot_settings["do_2d_plots"]
        self.do_3d_plots = plot_settings["do_3d_plots"]
        self.do_interactive_plots = plot_settings["do_interactive_plots"]
        self.pca = []
        self.processed_names = []
        self.name_mask = []


    def create_plots(self, pca, processed_names, name_mask):

        self.pca = pca
        self.processed_names = processed_names
        self.name_mask = name_mask
        print("Creating plots...")
        if self.do_2d_plots:
            self._create_2d_plots()
        if self.do_3d_plots:
            self._create_3d_plots()
        if self.do_interactive_plots:
            self._create_interactive_plot()



    def _create_2d_plots(self):
        plt.figure(figsize=(8, 6))
        unique_labels, numeric_label = np.unique(self.processed_names, return_inverse=True)
        sc = plt.scatter(self.pca[:, 0], self.pca[:, 2], c=numeric_label, cmap='hsv', s=50)
        # plt.legend(handles=sc.legend_elements()[0], labels=list(unique_labels))
        plt.colorbar(sc, label=self.processed_names)
        plt.xlabel('PC1')
        plt.ylabel('PC3')
        plt.title(f'{self.plot_name} 2D')
        plt.grid(True)
        plt.show()

    def _create_3d_plots(self):
        unique_labels, numeric_label = np.unique(self.processed_names, return_inverse=True)

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

        for i, name in enumerate(self.processed_names):
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
        ax.set_title(f'{self.plot_name} 3D')
        ax.legend()  # Show the label for the 'X' markers
        plt.show()


    def _create_interactive_plot(self):

        # 1. Setup Labels and Colors
        unique_labels, numeric_label = np.unique(self.processed_names, return_inverse=True)
        mask_array = np.array(self.name_mask, dtype=bool)

        # 3. Filter "Full Morphs" for labels (De-duplication by ROUNDED coordinate)
        mask_indices = np.where(mask_array)[0]
        seen_coords = set()

        final_mask_x = []
        final_mask_y = []
        final_mask_z = []
        final_mask_text = []

        for idx in mask_indices:
            # Rounding to 2 decimal places handles the "almost identical" issue
            # If your points are extremely close, try rounding to 1 or 3 instead
            coord = tuple(np.round(self.pca[idx], 2))

            if coord not in seen_coords:
                final_mask_x.append(self.pca[idx, 0])
                final_mask_y.append(self.pca[idx, 1])
                final_mask_z.append(self.pca[idx, 2])
                # Using the simplified processed name for the label
                final_mask_text.append(self.processed_names[idx])
                seen_coords.add(coord)

        # 4. Create the Figure
        fig = go.Figure()

        # Trace 1: All points (The background "cloud")
        fig.add_trace(go.Scatter3d(
            x=self.pca[:, 0], y=self.pca[:, 1], z=self.pca[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=numeric_label,
                colorscale='Hsv',
                opacity=0.4
            ),
            text=self.processed_names,  # Hover still shows original full filename
            name='All Images',
            hoverinfo='text'
        ))

        # Trace 2: Highlighted Full Morphs (Unique/Rounded locations only)
        fig.add_trace(go.Scatter3d(
            x=final_mask_x, y=final_mask_y, z=final_mask_z,
            mode='markers+text',
            marker=dict(size=8, color='black', symbol='x'),
            text=final_mask_text,  # Uses the cleaned "bark" style names
            textposition="top center",
            textfont=dict(size=10, color='black'),
            name='Unique Full Morphs',
            hoverinfo='text'
        ))

        # 5. Layout and Export
        fig.update_layout(
            template="plotly_white",
            title=f"{self.plot_name} 3D Projection",
            scene=dict(
                xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        filename = f"{self.plot_name}.html"
        fig.write_html(filename)


