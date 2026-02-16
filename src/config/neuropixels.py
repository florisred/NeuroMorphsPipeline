import numpy as np
from pathlib import Path
from os.path import join

class Neuropixels:

    def __init__(self):
        self.data_dir = ''
        self.population_matrix = []
        self.timestamps = []
        self.labels = []
        self.label_dict = {}


        pass

    def load_data(self):
        neuropixels_dir = join(self.data_dir, 'neuropixels_data')
        self.population_matrix = np.load(neuropixels_dir + 'population_matrix.npy')
        self.timestamps = np.load(neuropixels_dir + 'timestamps.npy')
        self.labels = np.load(neuropixels_dir + 'labels.npy')

    def process_neuropixels_data(self):


        self._match_timestamps()

        self._separate_population()



    def _match_timestamps(self):
        # labels is een df met de timestamps en de labels die erbijhoren
        # maak een dict met indeces van de population matrix van elke label
        pass

    def _separate_population(self):

        for label, indeces in self.label_dict.items():
            self.population_matrix[indeces] = label

