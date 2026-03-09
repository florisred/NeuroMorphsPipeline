from io_tools.io_funcs import choose_transitions
from abc import ABC
from helper.PCATools import PCATools

class PCAMixin(ABC, PCATools):

    def get_transformed_data(self, subset = 'full'):
        match subset:
            case 'full': transitions = None
            case 'triplets': transitions = self.partial_transitions[0]
            case 'quadruplets': transitions = self.partial_transitions[1]
            case 'choose': transitions = choose_transitions(self.labels)
            case _:
                if type(subset) == list: transitions = subset
                else:
                    raise ValueError(f"subset defined incorrectly, must be 'full', 'triplets', 'quadruplets' or a list of transitions, not '{subset}'")

        return self._perform_pca(
            data_df = self.data_df,
            labels = self.labels,
            transitions = transitions
        )

    @property
    def partial_transitions(self):
        """
        returns a list of triplets and quadruplets possible with the loaded data
        """
        triplets = self._find_triplets(self.labels)
        quadruplets = self._find_quadruplet_chains(self.labels)
        return triplets, quadruplets









