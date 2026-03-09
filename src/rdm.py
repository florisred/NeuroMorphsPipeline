import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns



class RepresentationalDissimilarityMatrix:


    def __init__(self, pca_dict):
        self.pca_dict = pca_dict


    def _sort_sessions(self):
        sessions = list(self.pca_dict.keys())

        data_sources = np.unique([session.split('_')[0] for session in sessions])
        test=1



    def run_rdm(self):
        self._sort_sessions()

        rdms = {}

        for session_id, session_tuple in self.pca_dict.items():
            centroids = []
            pca_data = session_tuple[0]
            dist_vector = pdist(np.array(pca_data), metric='euclidean')
            rdms[session_id] = dist_vector

        for key in rdms.keys():
            rdm_matrix = squareform(rdms[key])
            labels = self.pca_dict[key][2]['morph_name'].values
            plt.figure(figsize=(12, 10))
            ax = sns.heatmap(rdm_matrix, xticklabels=labels, yticklabels=labels)

            # if not all_labels:
            #     [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if "10" not in labels[i]]
            #     [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if "10" not in labels[i]]

            plt.title(f"Session {key} RDM")
            plt.show()