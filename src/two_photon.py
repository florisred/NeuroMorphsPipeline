from itertools import groupby
from os.path import join
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import itertools



class Twophoton:
    def __init__(self):
        self.data_df = None
        self.labels = None
        self.processed_df = None
        self.labels_list = ["y/pair_key", "y/step_index", "y/src_cat", "y/dst_cat", "y/norm_step", "y/stim_type"]

    def load_2p_data(self, session_dirs, data_location, label_location):
        """

        """
        data_dfs = []
        for session_dir in session_dirs:

            two_photon_folder = join(session_dir, "2p_data")
            file = list(Path(two_photon_folder).glob("*.h5"))
            if len(file) != 1: raise AssertionError(f"Need exactly one .hy file in {two_photon_folder}")
            f = h5py.File(file[0], 'r')
            data = pd.DataFrame(f[data_location]).dropna(axis=0, how='all')
            labels = self._load_list(f, self.labels_list)
            grouped_data, index_df = self._calc_mean_per_stimulus(data, labels)
            grouped_data.index = index_df['morph_name']
            data_dfs.append(grouped_data)

        # concatinate and scale
        combined_df = pd.concat(data_dfs, axis=1, join='inner')
        index_df= combined_df.index.to_frame(index=False, name='morph_name')
        labels_df = index_df.merge(labels.drop_duplicates('morph_name'), on='morph_name', how='left')


        scaler = StandardScaler()
        scaled_combined_df = pd.DataFrame(scaler.fit_transform(combined_df))
        self.data_df = scaled_combined_df
        self.labels = labels_df




    def perform_pca_subset(self, choose_transitions = False):
        """
        """
        if choose_transitions:
            triplets = [self._io_partial_pca()]
            quadruplets = []
        else:
            triplets = self._find_triplets()
            quadruplets = self._find_quadruplet_chains()
        pca_dict = {}

        for triplet in triplets:
            bases =  np.unique([trip.split("__") for trip in triplet])
            pca_name = f"triplet"
            for base in bases: pca_name += f'-{base}'
            pca_dict[pca_name] = self.perform_pca(
                transitions=triplet
            )

        for quadruplet in quadruplets:
            bases = np.unique([quad.split("__") for quad in quadruplet])
            pca_name = f"quadruplet"
            for base in bases: pca_name += f'-{base}'
            pca_dict[pca_name] = self.perform_pca(
                transitions=quadruplet
            )

        return pca_dict

    def _find_triplets(self):
        """
        Finds all possible triplets of unique stimuli where each triplet forms a complete triangle
        based on the transitions present in the dataset.

        This method analyzes the provided dataset for unique stimuli pairs and identifies all
        combinations of three stimuli that together form a triangle. A triangle is formed when
        all three possible edges between the stimuli exist in the dataset of transitions.

        :return: A list of tuples representing possible triplets of unique stimuli forming complete triangles.
                 Each tuple contains three edges, where each edge is represented as a string of two
                 sorted stimuli joined by '__'.
        :rtype: list[tuple[str, str, str]]
        """
        transitions = self.labels["pair_key"].dropna().values
        unique_stimuli = set()
        for t in transitions:
            unique_stimuli.update(t.split('__'))
        possible_triplets = []
        for a, b, c in itertools.combinations(unique_stimuli, 3):
            # Determine the three edges of the triangle (sorted to match your list)
            edge1 = "__".join(sorted([a, b]))
            edge2 = "__".join(sorted([b, c]))
            edge3 = "__".join(sorted([c, a]))
            # Check if all three edges exist in your dataset
            if all(edge in transitions for edge in [edge1, edge2, edge3]):
                possible_triplets.append((edge1, edge2, edge3))
        print(possible_triplets)
        return possible_triplets

    def _find_quadruplet_chains(self):
        """
        Finds and returns all possible quadruplet chains formed by a set of nodes, where each chain forms
        a single ring (cyclic path). The method identifies chains based on predefined edge relationships
        among the nodes.

        The algorithm works by:
        1. Standardizing the edges into a set of frozensets for fast lookup.
        2. Iterating over all combinations of four unique nodes and checking if they can form a valid ring.
        3. Defining three possible paths in which four nodes can form a single ring, ensuring each chain's
           edges match the predefined edge relationships.

        :rtype: list[list[str]]
        :return: A list of quadruplet chains, where each chain is a list of four strings, each representing
                 a standardized edge in the form of "node1__node2". Each chain corresponds to a ring structure.
        """
        # 1. Standardize the edges into a set of frozensets for fast lookup
        edges_set = set()
        source_cat = self.labels["src_cat"].tolist()
        dst_cat = self.labels["dst_cat"].tolist()
        for src, dst in zip(source_cat, dst_cat):
            edges_set.add(frozenset((src, dst)))

        unique_stimuli = list(set().union(*edges_set))
        found_chains_as_labels = []

        # 2. Check combinations of 4 nodes
        for nodes in itertools.combinations(unique_stimuli, 4):
            a, b, c, d = nodes

            # Define the 3 possible ways 4 nodes can form a single ring
            possible_paths = [
                (a, b, c, d),
                (a, b, d, c),
                (a, c, b, d)
            ]

            for p in possible_paths:
                # Create the 4 specific "hops" for this path
                # We sort them so 'brick__bark' always becomes 'bark__brick' to match your data
                hops = [
                    "__".join(sorted([p[0], p[1]])),
                    "__".join(sorted([p[1], p[2]])),
                    "__".join(sorted([p[2], p[3]])),
                    "__".join(sorted([p[3], p[0]]))
                ]

                # Check if every "hop" in this chain exists in our edge set
                if all(frozenset(h.split('__')) in edges_set for h in hops):
                    found_chains_as_labels.append(hops)

        return found_chains_as_labels

    def _io_partial_pca(self):
        unique_trans = np.unique(self.labels["pair_key"])
        chosen_transitions = []
        print("Possible transisitions:")
        for trans in unique_trans:
            print(trans)
        while True:
            print(f"Current list: {chosen_transitions}")
            chosen_trans = input("Add or remove transitions, type q to stop:")
            if chosen_trans == "q": break
            elif chosen_trans not in chosen_transitions:
                if chosen_trans not in unique_trans:
                    print("please choose a valid transition")
                chosen_transitions.append(chosen_trans)
            elif chosen_trans in chosen_transitions: chosen_transitions.remove(chosen_trans)
            else:
                print("how did you get here?? I'm gonna crash because this shouldn't be possible. congrats!")
                raise AssertionError("what the hell")
        return chosen_transitions

    def _calc_mean_per_stimulus(self, data_df=None, labels=None):
        if labels is None:
            labels = self.labels
        if data_df is None:
            data_df = self.data_df

        grouped_data = data_df.T.groupby(labels["morph_name"].values).mean()
        grouped_data_labels = labels.drop_duplicates(subset='morph_name').sort_values(by=['morph_name'])
        return grouped_data, grouped_data_labels


    def perform_pca(self, transitions = None):
        """"
        Main function of this class. It takes the grouped activation data and the corresponding labels. It then calculates
        the so-called anchor points of the stimuli, and adds these. These are the average activation of the full morphs.

        It then creates a PCA space from only these anchor points.
        It then calculates the coordinates of all activation data based on this PCA space, and returns it.
        """

        labels = self.labels
        data_df = self.data_df

        if transitions is not None:
            data_df, labels = self._filter_transitions(
                data_df=data_df,
                labels = labels,
                transitions=transitions
            )

        anchors = self._extract_anchors(
            data_df = data_df,
            labels = labels
        )
        pca = PCA(n_components=(min(len(anchors), 4)))
        te_stankie = pca.fit_transform(anchors)
        variance_explained = pca.explained_variance_
        transformed_data = pca.transform(data_df)

        return transformed_data, variance_explained, labels

    @staticmethod
    def _filter_transitions(data_df, labels, transitions):
        mask1 = labels["pair_key"].isin(transitions)  # get all the data from the chosen transitions
        # find all the anchors
        relevant_anchors = np.unique(
            [texture for transition in transitions for texture in transition.split('__')])
        mask_relevant_morphs = labels['morph_name'].isin(relevant_anchors)
        # add both masks and filter data
        final_mask = mask1 | mask_relevant_morphs
        filtered_data = data_df[final_mask]
        filtered_labels = labels[final_mask]
        labels = filtered_labels
        return filtered_data, labels


    @staticmethod
    def _extract_anchors(data_df, labels):
        """"
        """
        is_anchor = labels['stim_type'] == 'anchor'
        anchor_data = data_df[is_anchor]
        return anchor_data



    @staticmethod
    def _load_list(f, metadata_list):
        metadata_dataframe = pd.DataFrame()
        for metadata_location in metadata_list:
            meta_name = metadata_location.split("/")[-1]
            metadata_array = np.array(f[metadata_location]).flatten()
            metadata_dataframe[meta_name] = metadata_array.astype(str)
        metadata_dataframe = metadata_dataframe.T
        metadata_df = metadata_dataframe.T
        is_src = metadata_df['stim_type'] == 'endpoint_src'
        is_dst = metadata_df['stim_type'] == 'endpoint_dst'
        partial_names = (
                metadata_df['src_cat'] + "_" +
                (1-metadata_df['norm_step'].astype(float)).round(2).astype(str) + "_" +
                metadata_df['dst_cat'] + "_" +
                metadata_df['norm_step'].astype(float).round(2).astype(str)
        )
        metadata_df['morph_name'] = np.where(
            is_src,
            metadata_df['src_cat'],
            np.where(is_dst, metadata_df['dst_cat'], partial_names)
        )
        metadata_df['stim_type'] = np.where(
            is_src | is_dst,
            'anchor',
            'morph'
        )
        metadata_df.loc[metadata_df['stim_type'] == 'anchor', 'pair_key'] = np.nan
        return metadata_df

