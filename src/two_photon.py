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
        self.pca_coords=None
        self.variance_explained=None
        self.metadata_list = [
                "meta/mouse",
                "meta/date",
                "meta/sess_repN",
                "meta/sess_index"
            ]
        self.labels_list = ["y/pair_key", "y/step_index", "y/src_cat", "y/dst_cat", "y/norm_step", "y/stim_type"]
        # self.partial_pca_full_morphs = None
        # self.partial_pca_full_morphs_labels = None





    def load_2p_data(self, data_dir, data_location, label_location):
        """
        Loads 2-photon imaging data from the specified directory and file locations. The function retrieves the data
        and labels, applies necessary preprocessing, and stores them into appropriate attributes for further use.

        :param data_dir: Directory containing the 2-photon imaging data files.
        :type data_dir: str
        :param data_location: Key or path specifying the location of the dataset in the .h5 file.
        :type data_location: str
        :param label_location: Key or path specifying the location of the labels in the .h5 file.
        :type label_location: str
        :return: None
        """
        two_photon_folder = join(data_dir, "2p_data")
        file = list(Path(two_photon_folder).glob("*.h5"))
        if len(file) != 1: raise AssertionError("Need exactly one .hy file")
        file = file[0]
        f = h5py.File(file, 'r')
        self.data_df = pd.DataFrame(f[data_location]).dropna(axis=0, how='all')
        self.labels = self._load_list(f, self.labels_list)



    def partial_pca_full_morphs(self, choose_transitions = False, ):
        """
        Executes a partial Principal Component Analysis (PCA) on dataset transitions, either using specific
        transitions or clustered quadruplet chains. The method processes data and labels related to morphologies,
        combines them as needed, and performs the analysis to generate results grouped by unique morphologies.

        :param choose_transitions: Indicates whether to use specific transitions (if True) or acquire them
            through clustered quadruplet chains (if False).
        :type choose_transitions: bool

        :return: A dictionary where keys represent unique morphologies as a concatenated string of identifiers,
            and values are the results of partial PCA analysis for corresponding sets of transitions.
        :rtype: dict
        """
        if choose_transitions:
            triplets = [self._io_partial_pca()]
            quadruplets = []
        else:
            triplets = self._find_triplets()
            quadruplets = self._find_quadruplet_chains()
        pca_dict = {}
        for triplet in triplets:
            data_chosen_full_morphs, labels_chosen_full_morphs = self._filter_transitions_full_chosen_morphs(chosen_transitions=triplet)
            data_chosen_morphs, labels_chosen_morphs = self._filter_transitions_all_chosen_morphs(chosen_transitions=triplet)
            data_chosen_morphs, labels_chosen_morphs = self._concat_all_morphs(data_chosen_full_morphs, labels_chosen_full_morphs, data_chosen_morphs, labels_chosen_morphs)
            all_morphs = []
            for trans in triplet:
                all_morphs.append(trans.split("__"))
            all_morphs_unique = np.unique(all_morphs)
            pca_name = ""
            for name in all_morphs_unique:
                pca_name += name
            pca_dict[pca_name] = self._perform_partial_pca(data_chosen_full_morphs, labels_chosen_full_morphs, data_chosen_morphs, labels_chosen_morphs)

        for quadruplet in quadruplets:
            data_chosen_full_morphs, labels_chosen_full_morphs = self._filter_transitions_full_chosen_morphs(chosen_transitions=quadruplet)
            data_chosen_morphs, labels_chosen_morphs = self._filter_transitions_all_chosen_morphs(chosen_transitions=quadruplet)
            data_chosen_morphs, labels_chosen_morphs = self._concat_all_morphs(data_chosen_full_morphs, labels_chosen_full_morphs, data_chosen_morphs, labels_chosen_morphs)
            all_morphs = []
            for trans in quadruplet:
                all_morphs.append(trans.split("__"))
            all_morphs_unique = np.unique(all_morphs)
            pca_name = ""
            for name in all_morphs_unique:
                pca_name = pca_name + '-' + name
            pca_dict[pca_name] = self._perform_partial_pca(data_chosen_full_morphs, labels_chosen_full_morphs, data_chosen_morphs, labels_chosen_morphs)

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
        transitions = self.labels["pair_key"].values
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

    import itertools

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

    def _concat_all_morphs(self, data_chosen_full_morphs, labels_chosen_full_morphs, data_chosen_morphs, labels_chosen_morphs):
        """
        Concatenates multiple datasets of morphological data and their corresponding labels into
        a unified dataset and label set. This function ensures that the resulting datasets retain
        consistent indexing.

        :param data_chosen_full_morphs: A pandas DataFrame containing a dataset of full morphological data.
        :param labels_chosen_full_morphs: A pandas DataFrame containing the corresponding labels for
            the data in `data_chosen_full_morphs`.
        :param data_chosen_morphs: A pandas DataFrame containing another dataset of selected
            morphological data to be concatenated.
        :param labels_chosen_morphs: A pandas DataFrame containing the corresponding labels for
            the data in `data_chosen_morphs`.
        :return: A tuple with two pandas DataFrames:
            - The first element contains the combined dataset (`concated_data`).
            - The second element contains the combined labels (`concated_labels`).
        """

        concated_data = pd.concat([data_chosen_full_morphs, data_chosen_morphs]).reset_index(drop=True)
        concated_labels = pd.concat([labels_chosen_full_morphs, labels_chosen_morphs]).reset_index(drop=True)
        return concated_data, concated_labels

    def _perform_partial_pca(self, data_chosen_full_morphs, labels_chosen_full_morphs, data_chosen_morphs, labels_chosen_morphs):
        """
        Performs a partial PCA (Principal Component Analysis) transformation on the given dataset.

        The method applies PCA to the subsets of provided data and computes the PCA coordinates
        and the explained variance. The labels corresponding to the chosen morphs are
        returned as-is.

        :param data_chosen_full_morphs: Data representing the full set of morph features to be
            considered in the PCA transformation.
        :param labels_chosen_full_morphs: Labels corresponding to the full morph dataset, used
            for reference or further processing after the PCA transformation.
        :param data_chosen_morphs: Subset of the morph feature data to use for computing
            partial PCA. Represents a selected portion of the data.
        :param labels_chosen_morphs: Labels corresponding to the subset of morph data. These
            labels are returned alongside the PCA results for context.

        :return: A tuple containing:
            - `pca_coords`: The computed PCA coordinates for the given data subsets.
            - `variance_explained`: The proportion of variance explained by each principal
              component.
            - `labels_chosen_morphs`: The same labels provided for the chosen morphs,
              returned for reference.
        """
        pca_coords, variance_explained, _ = self.peform_pca(data_chosen_full_morphs, data_chosen_morphs)
        return pca_coords, variance_explained, labels_chosen_morphs


    def _filter_transitions_all_chosen_morphs(self, chosen_transitions):
        """
        Filters, scales, and groups activation data based on transitions and suffixes.

        This method processes transitions combined with suffixes to create specific
        targets. Matching data is then filtered, grouped by certain labels, and
        averaged. The resulting activation data is standardized using a scaler.
        It returns the scaled mean activations and the corresponding grouped labels.

        :param chosen_transitions: A list of transition strings to be combined with
            suffixes for filtering targets.
        :type chosen_transitions: list of str
        :return: A tuple containing the scaled mean activations as a DataFrame and
            the grouped labels as a Series.
        :rtype: tuple (pd.DataFrame, pd.Series)
        """

        mask = self.labels["pair_key"].isin(chosen_transitions)
        data_masked = self.data_df.T[mask]
        labels_masked = self.labels[mask]

        groups = labels_masked['full_name'].squeeze()
        mean_activations = data_masked.groupby(groups).mean()
        scaler = StandardScaler()
        mean_activations_scaled = scaler.fit_transform(mean_activations)
        labels_grouped  = mean_activations.index.to_series().reset_index(drop=True)
        mean_activations_scaled = pd.DataFrame(mean_activations_scaled)

        #prepare the labels df for return
        labels_unique = self.labels.drop_duplicates(subset=['full_name'])
        labels_return = pd.DataFrame(labels_grouped).merge(labels_unique, on='full_name', how='left')


        return mean_activations_scaled, labels_return

    def _filter_transitions_full_chosen_morphs(self, chosen_transitions):

        ## get the three starting points
        min_morph = 0
        max_morph = np.max(self.labels["step_index"].astype(int))
        suffixes = [f"{min_morph:02d}", f"{max_morph:02d}"]
        targets = [trans + surf for trans in chosen_transitions for surf in suffixes]


        mask = self.labels["full_name"].isin(targets)
        data_masked = self.data_df.T[mask]
        labels_masked = self.labels[mask]["full_name"].apply(
            lambda x: x.split('__')[0] if x.endswith('00') else x.split('__')[1][:-2])
        groups = labels_masked.squeeze()
        mean_activations = data_masked.groupby(groups).mean()
        scaler = StandardScaler()
        mean_activations_scaled = scaler.fit_transform(mean_activations)
        mean_activations_scaled = pd.DataFrame(mean_activations_scaled)
        labels_grouped  = mean_activations.index.to_series().reset_index(drop=True)
        #prepare the labels df for return
        labels_unique = self.labels.drop_duplicates(subset=['full_name'])
        labels_return = pd.DataFrame(labels_grouped).merge(labels_unique, on='full_name', how='left')
        labels_return['stim_type'] = 'anchor'

        return mean_activations_scaled,labels_return

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




    def _calc_mean_per_stimulus_and_scale(self):
        processed_df = self.data_df.T.groupby(self.labels["full_name"].values).mean()
        scaler = StandardScaler()
        processed_df = scaler.fit_transform(processed_df)
        self.processed_df = processed_df



    def peform_pca(self, data_full_morphs=None, data_all=None):
        self._calc_mean_per_stimulus_and_scale()


        if data_full_morphs is None and data_all is None:
            data = self.processed_df
            pca = PCA(n_components=(min(len(data), 8)))
            pca_coords = pca.fit_transform(data)
            variance_explained = pca.explained_variance_ratio_
            labels_unique = self.labels.drop_duplicates(subset=['full_name'])
            return pca_coords, variance_explained, labels_unique
        else:
            pca = PCA(n_components=(min(len(data_full_morphs), 8)))
            pca_coords = pca.fit_transform(data_full_morphs)
            variance_explained = pca.explained_variance_ratio_
            pca_coords_all = pca.transform(data_all)
            labels_unique = self.labels.drop_duplicates(subset=['full_name'])

            return pca_coords_all, variance_explained, labels_unique



    @staticmethod
    def _load_list(f, metadata_list):
        metadata_dataframe = pd.DataFrame()
        for metadata_location in metadata_list:
            meta_name = metadata_location.split("/")[-1]
            metadata_array = np.array(f[metadata_location]).flatten()
            metadata_dataframe[meta_name] = metadata_array.astype(str)
        metadata_dataframe = metadata_dataframe.T
        metadata_df = metadata_dataframe.T
        col1 = metadata_df.iloc[:, 0].astype(str)
        col2 = metadata_df.iloc[:, 1].astype(str).str.zfill(2)
        metadata_df['full_name'] = col1 + col2
        return metadata_df


    ## krijg alle data in aparte columns
    # codeer het zodat elke functie daaruit haalt
    # verwijder "full_name"