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
    """
    The Blueprint for each datasource object. 
    """
    def __init__(
            self,
            file_paths: list[Path],
    ):
        self.file_paths = file_paths
        self._data = None
        self._metadata = TrialMetadata()
        self._data_type = 'Unknown'
        self._use_train_test_all = 'all'
        self._train_mask = None
        self._filter_mask = None
        self._use_mask = False
        self._split = False

    @property
    def data_type(self) -> str:
        """
        The data type returns a string that contains the information about this data source. For example, two_photon, 
        Gabor, Pixel, etc. If it contains split data (train-test), add that too.
        """
        
        if self.is_split:
            return f'{self._data_type}split'
        else: return self._data_type

    @property
    def is_split(self):
        return self._split


    @abstractmethod
    def load_data(self):
        """
        Loads data from file
        """
        pass

    @property
    def anchors(self) -> pd.DataFrame:
        """
        Retrieves the data of all the anchors and returns it
        :return: a DataFrame of the anchor data
        """
        return self.data[self.metadata.anchor_mask]

    @property
    def data(self):
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

    @property
    def metadata(self):
        return self._metadata.copy()

    def copy(self):
        return copy.deepcopy(self)

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

    def filter_transitions(self, transitions: list[str]):
        """
        applies a mask to the data, such that only data with of these pair__types become visible
        :param transitions: a list of transitions (pair__types)
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
        self._metadata.apply_mask(final_mask)
        self._use_mask = True

    def set_train_test(self, to_use: str):
        """
        Applies a mask to the data, such that only train or test data becomes visible.
        :param to_use: a string of what to use (train, test, or all)
        """
        if to_use not in ['test', 'train', 'all']:
            logger.error('Use either test, train, or all')
        self._use_train_test_all = to_use
        if to_use == 'all':
            self._train_mask = [True for i in range(len(self._data))]
        else:
            self._train_mask = self._data.index.str.endswith(to_use)
        self._metadata.apply_train_mask(self._train_mask)
        self._use_mask = True



