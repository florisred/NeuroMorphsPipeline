import itertools
import numpy as np
from sklearn.decomposition import PCA

class PCATools:

    def _perform_pca(self, data_df, labels, transitions = None):

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
    def _find_quadruplet_chains(labels):
        edges_set = set()
        source_cat = labels["src_cat"].tolist()
        dst_cat = labels["dst_cat"].tolist()
        for src, dst in zip(source_cat, dst_cat):
            edges_set.add(frozenset((src, dst)))
        unique_stimuli = list(set().union(*edges_set))
        found_chains_as_labels = []
        for nodes in itertools.combinations(unique_stimuli, 4):
            a, b, c, d = nodes
            possible_paths = [
                (a, b, c, d),
                (a, b, d, c),
                (a, c, b, d)
            ]
            for p in possible_paths:
                hops = [
                    "__".join(sorted([p[0], p[1]])),
                    "__".join(sorted([p[1], p[2]])),
                    "__".join(sorted([p[2], p[3]])),
                    "__".join(sorted([p[3], p[0]]))
                ]
                if all(frozenset(h.split('__')) in edges_set for h in hops):
                    found_chains_as_labels.append(hops)

        return found_chains_as_labels

    @staticmethod
    def _find_triplets(labels):
        transitions = labels["pair_key"].dropna().values
        unique_stimuli = set()
        for t in transitions:
            unique_stimuli.update(t.split('__'))
        possible_triplets = []
        for a, b, c in itertools.combinations(unique_stimuli, 3):
            edge1 = "__".join(sorted([a, b]))
            edge2 = "__".join(sorted([b, c]))
            edge3 = "__".join(sorted([c, a]))
            if all(edge in transitions for edge in [edge1, edge2, edge3]):
                possible_triplets.append((edge1, edge2, edge3))
        print(possible_triplets)
        return possible_triplets

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
        is_anchor = labels['stim_type'] == 'anchor'
        anchor_data = data_df[is_anchor]
        return anchor_data
