# NeuroMorphsPipeline

### What is it?
The neuromorphs pipeline is a small python pipeline for the analysis of 2-photon and neuropixels data.
It is primarily designed to analyze the responses of V1 after showing certain stimuli, in this case texture morphs.
This was created by me, Floris Hoogenboom, as part of my internship at the Netherlands institute for Neuroscience. 

### How does it work?
In the config file, point to a folder containing different sessions you want to analyze. Make sure each session you want to analyze either a folder called '2p_data', or 'neuropixels_data'.
This program is able to analyze and perform PCA on the image files (the stimuli), neuropixels data, and two-photon data.
In order to determine how the stimuli relate to the neural responses, we use PCA on both the image data and the neural data. 


### Folder structure
-- root_folder

-----> stimuli

-----> session_folder_1

-----------> 2p_data

-----> session_folder_2

-----------> neuropixels_data


### Debug stuff
Basically, run the run_pipeline.py. This will initiate a Pipeline() object. 
This object gets information from config/settings.json
Then, the script will check in the session folder if there is a 2p, ephys, and stimuli folder. It will then initilize 
DataSource() objects, like TwoPhotonDataSource(). These all do the same thing, but load the data in a slightly different manner.
Each DataSource() object has a TrialMetaData() object attached to it, which serves as an easy way to access and manipulate 
the metadata. 

For the analysis, we pass all the DataSource() objects to an Analyzer() object, which has MixIns to perform
both PCA and create the necessary plots. This way, if you ever need to make a new type of plot, this is quite easy.