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
        self._filter_mask = None
        self._use_mask = False

    @property
    def data_type(self) -> str:
        """
        The data type returns a string that contains the information about this data source. For example, two_photon, 
        Gabor, Pixel, etc. If it contains split data (train-test), add that too.
        """

        return self._data_type


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
        mask_relevant_anchors = [name in relevant_anchors for name in all_morph_names]
        final_mask = mask1 | np.array(mask_relevant_anchors)
        self._metadata.apply_mask(final_mask)
        self._use_mask = True

    def update_data_source(self, name: str, append:bool=True):
        if append:
            prev_nm = self.data_type
            self._data_type = prev_nm + name
        else:
            self._data_type = name





