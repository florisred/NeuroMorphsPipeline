from abc import ABC, abstractmethod
from pathlib import Path
import itertools
from data_objects.trial_metadata import TrialMetadata
import numpy as np
import copy

class DataSource(ABC):
    def __init__(
            self,
            file_paths: list[Path],
    ):
        self.file_paths = file_paths
        self.data = None
        self.metadata = TrialMetadata()
        self._use_mask = False
        self.mask = None
        self.data_type = 'Unknown'

    def get_data_type(self) -> str:
        return self.data_type

    @abstractmethod
    def load_data(self):
        """
        Loads data from file
        """
        pass

    def find_stimulus_cycles(self, n=3):
        """
        Finds all unique n-length cycles (triplets, quadruplets, etc.) in the stimulus transitions.

        :param labels: The metadata DataFrame/dict containing transitions.
        :param n: The number of nodes in the cycle (3 for triangles, 4 for rings).
        :return: A list of lists, where each inner list contains the edge keys forming the cycle.
        """
        # 1. Standardize edges into a set of frozensets for O(1) lookup
        # This replaces the slow 'if edge in transitions' check
        edges_set = set()

        for t in self.metadata.get_pair_keys():
            edges_set.add(frozenset(t.split("__")))

        unique_nodes = list(set().union(*edges_set))
        found_cycles = []

        for nodes in itertools.combinations(unique_nodes, n):
            for p in itertools.permutations(nodes[1:]):
                path = (nodes[0],) + p
                valid_cycle = True
                current_hops = []
                for i in range(n):
                    u, v = path[i], path[(i + 1) % n]
                    edge = frozenset((u, v))

                    if edge in edges_set:
                        current_hops.append("__".join(sorted([u, v])))
                    else:
                        valid_cycle = False
                        break
                if valid_cycle:
                    cycle_id = tuple(sorted(current_hops))
                    if cycle_id not in [tuple(sorted(c)) for c in found_cycles]:
                        found_cycles.append(current_hops)

        return found_cycles



    def filter_transitions(self, transitions: list[str]):
        """
        applies a mask to the
        """
        mask1 = self.metadata.get_pair_keys(
            unique=False,
            dropna=False,
        ).isin(transitions)
        relevant_anchors = np.unique(
            [texture for transition in transitions for texture in transition.split('__')])
        mask_relevant_morphs = self.metadata.get_morph_names().isin(relevant_anchors)
        final_mask = mask1 | mask_relevant_morphs
        self.mask = final_mask
        self.metadata.apply_mask(final_mask)
        self._use_mask = True


    def get_anchors(self):
        mask = self.metadata.get_anchor_mask()
        filtered_data = self.get_data()[mask]
        return filtered_data

    def get_data(self):
        if self._use_mask:
            return self.data[self.mask]
        return self.data

    def get_metadata(self):
        return self.metadata

    def copy(self):
        return copy.deepcopy(self)




