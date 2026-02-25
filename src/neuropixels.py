import numpy as np
from src.nidq import Nidq
from pathlib import Path
from os.path import join
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class NeuroPixelsData:

    def __init__(self):
        self.population_matrix = None
        self.data_dir = ''
        self.timestamps = []
        self.labels = []
        self.label_dict = {}
        self.trial_duration = 0.3 #seconds
        self.time_bin_duration = 0.02 # seconds
        self.df = pd.DataFrame()
        self.pca_data = None
        self.final_df = None
        self.stimulus = None
        self.te_stankie = None

    def load_data(self, data_dir):
        self.data_dir = data_dir
        neuropixels_dir = join(data_dir, 'neuropixels_data')
        self.population_matrix = np.load(join(neuropixels_dir, 'population_matrix.npy'))

        self.timestamps = np.load(join(neuropixels_dir, 'timestamps.npy'))
        #self.labels = np.load(join(neuropixels_dir, 'labels.nidq.bin'))
        nidq = Nidq(neuropixels_dir)
        nidq.load_data()
        self.stimulus = nidq.data_df_processed["stimulus"]
        test = 1


    def process_neuropixels_data(self):

        self._calc_avg_bin()
        self._process_pca()
        return self.te_stankie, self.pca_data, self.final_df.index.tolist()


    def _calc_avg_bin(self):

        popmat = pd.DataFrame(self.population_matrix)
        popmat["stimulus"] = self.stimulus
        popmat = popmat.replace("", np.nan).dropna(subset=["stimulus"])
        popmat['bin_index'] = popmat.groupby((popmat['stimulus'] != popmat['stimulus'].shift()).cumsum()).cumcount()
        neuron_cols = [c for c in popmat.columns if c not in ["stimulus", "bin_index"]]
        avg_popmat = popmat.groupby(['stimulus', 'bin_index'])[neuron_cols].mean().reset_index()
        # 2. Now pivot will work because each stimulus has exactly one bin_index 0, 1, 2...
        flattened_df = avg_popmat.pivot(index='stimulus', columns='bin_index', values=neuron_cols)

        # 3. Clean up columns
        flattened_df.columns = [f"cell{c}_bin{b}" for c, b in flattened_df.columns]
        self.final_df = flattened_df

    def _process_pca(self):
        # 1. Initialize the Scaler
        scaler = StandardScaler()

        # 2. Scale the data
        # StandardScaler returns a NumPy array, so we lose the index/columns temporarily
        df_scaled = scaler.fit_transform(self.final_df)

        # 3. Perform PCA on the scaled data
        te_stankie = PCA(n_components=8)
        pca_results = te_stankie.fit_transform(df_scaled)

        # 4. Reconstruct as a DataFrame to keep your 'stimulus' index
        self.pca_data = pd.DataFrame(
            pca_results,
            index=self.final_df.index,
            columns=[f'PC{i + 1}' for i in range(8)]
        )

        # 5. Save the objects for later use
        self.te_stankie = te_stankie
        self.scaler = scaler

