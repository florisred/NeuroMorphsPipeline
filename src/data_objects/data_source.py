from abc import ABC, abstractmethod
from pathlib import Path
import itertools

import pandas as pd

from data_objects.trial_metadata import TrialMetadata
import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)

class DataSource(ABC):
    def __init__(
            self,
            file_paths: list[Path],
    ):
        self.file_paths = file_paths
        self._data = None
        self._data_raw = None
        self.metadata = TrialMetadata()
        self.mask = None
        self.data_type = 'Unknown'
        self._use_train_test_all = 'all'
        self._train_mask = None
        self._filter_mask = None
        self._use_mask = False
        self._split = False


    def get_data_type(self) -> str:
        if self.is_split:
            return f'{self.data_type}split'
        else: return self.data_type

    def train_test_mask(self, to_use: str):
        if to_use not in ['test', 'train', 'all']:
            logger.error('Use either test, train, or all')
        self._use_train_test_all = to_use
        if to_use == 'all':
            self._train_mask = [True for i in range(len(self._data))]
        else:
            self._train_mask = self._data.index.str.endswith(to_use)
        self.metadata.apply_train_mask(self._train_mask)
        self._use_mask = True

    def filter_transitions(self, transitions: list[str]):
        """
        applies a mask to the
        """
        if type(transitions) is not list: raise TypeError('transitions should be a list')
        mask1 = self.metadata.get_pair_keys(
            unique=False,
            dropna=False,
        ).isin(transitions)
        relevant_anchors = np.unique(
            [texture for transition in transitions for texture in transition.split('__')])
        all_morph_names =  self.metadata.get_morph_names(ignore_mask=True)
        if self._split:
            all_morph_names = [name.split('_')[-2] for name in all_morph_names]
        mask_relevant_anchors = [name in relevant_anchors for name in all_morph_names]
        final_mask = mask1 | np.array(mask_relevant_anchors)
        if self._train_mask is not None: final_mask &= self._train_mask

        self._train_mask = final_mask
        self.metadata.apply_mask(final_mask)
        self._use_mask = True

    @property
    def is_split(self):
        return self._split


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


    def get_anchors(self):
        mask = self.metadata.get_anchor_mask()
        filtered_data = self.get_data()[mask]
        return filtered_data

    def get_data(self):
        if self._use_mask:
            if self._train_mask is not None and self._filter_mask is not None:
                mask = self._train_mask & self._filter_mask
            elif self._train_mask is not None:
                mask = self._train_mask
            elif self._filter_mask is not None:
                mask = self._filter_mask
            else:
                logger.error("Somehow, mask was activated without either a train_mask or a filter_mask. Check what happened!")
                mask = [True for i in range(len(self._data))]
            return self._data[mask]

        return self._data

    def get_metadata(self):
        return self.metadata.copy()

    def copy(self):
        return copy.deepcopy(self)





## ToDo: Add a disable mask option