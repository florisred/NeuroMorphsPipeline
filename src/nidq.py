import numpy as np
from pathlib import Path
from os.path import join
import spikeinterface.extractors as se
import pandas as pd
from scipy.io import loadmat


class Nidq:

    ## psuedo
    # label_data = load_.nidq.bin
    # extract clock data and stimulus data
    # convert to seconds and add as an extra column
    # align to neuropixel data
    # find edges of stim data -> save index

    def __init__(self,
                 neuropixels_data_dir,

    ):
        self.neuropixels_data_dir = neuropixels_data_dir
        self.timestamps = None
        self.sampling_freq = None
        self.data = None
        self.data_df = None
        self.data_df_processed = None
        self.stimuli_df = None


    def load_data(self):
        self.timestamps = np.load(join(self.neuropixels_data_dir, 'timestamps.npy'))
        file = list(Path(self.neuropixels_data_dir).glob("*.nidq.bin"))
        if len(file) != 1: raise AssertionError("Need exactly one .nidq.bin file")
        recording = se.read_spikeglx(self.neuropixels_data_dir, stream_id='nidq')
        channel_ids = recording.channel_ids
        correct_channels = [channel_ids[0], channel_ids[-1]]
        recording_correct_channels = recording.select_channels(channel_ids=correct_channels)
        stimuli_file = list(Path(self.neuropixels_data_dir).glob("*.csv"))
        if len(stimuli_file) != 1: raise AssertionError("Need exactly one .csv file")
        stimulus_file = stimuli_file[0]
        self.stimuli_df = pd.read_csv(stimulus_file)
        self.stimuli_df["step_index"] = self.stimuli_df["step_index"].astype(str).str.zfill(2)
        self.sampling_freq = recording_correct_channels.sampling_frequency
        self.data_df = pd.DataFrame(recording_correct_channels.get_traces())
        self.convert_to_seconds()
        self.align_data()
        self.find_edges()
        test = 1


    def convert_to_seconds(self):
        data_df = self.data_df
        sampling_freq = self.sampling_freq
        length = len(data_df[0])
        seconds = [i* (1/sampling_freq) for i in range(length)]
        data_df["seconds"] = seconds
        self.data_df = data_df


    def align_data(self):
        data_df = self.data_df
        interp = np.interp(self.timestamps, data_df["seconds"], data_df[0])
        data_df_processed = pd.DataFrame()
        data_df_processed["volts"] = interp
        data_df_processed["time"] = self.timestamps
        self.data_df_processed = data_df_processed

    def find_edges(self):
        orit = loadmat(
            r"E:\Nextcloud\Documents\Education\VU\Internships\Final Year Project\shared\input\NeuroPixel_Logs\OriTuning_20251112_X_2.mat")
        sparse = loadmat(
            r"E:\Nextcloud\Documents\Education\VU\Internships\Final Year Project\shared\input\NeuroPixel_Logs\SparseNoise_20251112_X_1.mat")
        tot = int(orit["ntrials"][0][0]) + int(sparse["ntrials"][0][0])
        start_threshold = -5000
        end_threshold = -1000
        volts = self.data_df_processed["volts"].values
        is_stim = np.zeros(len(volts), dtype=bool)
        current_state = False
        for i in range(len(volts)):
            if not current_state:
                if volts[i] < start_threshold:
                    current_state = True
            else:
                if volts[i] > end_threshold:
                    current_state = False
            is_stim[i] = current_state
        self.data_df_processed["stim"] = is_stim
        self.data_df_processed = self.data_df_processed.drop(columns=["volts"])
        self.data_df_processed["stimulus"] = ""
        stim_state = self.data_df_processed["stim"]
        stim_starts = stim_state & ~stim_state.shift(1, fill_value=False)
        stim_block_ids = stim_starts.cumsum()
        experimental_mask = (stim_state) & (stim_block_ids >= tot)
        active_indices = self.data_df_processed.index[experimental_mask]
        active_blocks = stim_block_ids.loc[active_indices]

        if not active_blocks.empty:
            block_groups = active_blocks.groupby(active_blocks)
            min_length = block_groups.size().min()
            local_indices = block_groups.cumcount()
            print(f"Ignoring blocks 1 to {tot - 1}. "
                  f"Trimming experimental blocks ({len(block_groups)} total) to {min_length} bins.")
            mapping = {i + tot: f"{row['pair_id']}{row['step_index']}"
                       for i, (_, row) in enumerate(self.stimuli_df.iterrows())}
            trim_mask = local_indices < min_length
            final_keep_indices = active_indices[trim_mask]
            self.data_df_processed.loc[final_keep_indices, "stimulus"] = \
                active_blocks.loc[final_keep_indices].map(mapping)
        else:
            print(f"No blocks found with ID >= {tot}")

