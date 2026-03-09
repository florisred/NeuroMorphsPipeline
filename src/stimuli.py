from helper.StimuliHelper import StimuliHelper
from pathlib import Path
import pandas as pd
from MixIns.pcaMixin import PCAMixin
class Stimuli(PCAMixin, StimuliHelper):
    """"
    Class that handles the stimuli data
    """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dict = None
        self.data_type = 'gabor'

    def set_analysis_mode(self, mode):
        if mode not in ['gabor', 'pixel']: raise ValueError("mode must be either 'gabor' or 'pixel'")
        self.data_type = mode


    @property
    def labels(self):
        if self.data_dict is None: raise ValueError("stimuli metadata not loaded yet")
        return self.data_dict[self.data_type]['labels']


    @property
    def data_df(self):
        if self.data_dict is None: raise ValueError("stimuli data not loaded yet")
        return self.data_dict[self.data_type]['data']


    def process_images(self, gabor_params, return_dict = False):
        images, images_flat_df, stimuli_metadata = self._load_images()
        gabor_bank_df  = self._process_gabor(
            images=images,
            gabor_params=gabor_params
        )
        self._create_data_dict(
            images_flat = images_flat_df,
            gabor_bank = gabor_bank_df,
            stimuli_metadata = stimuli_metadata
        )

        if return_dict:
            return self.data_dict



