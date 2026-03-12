# NeuroMorphsPipeline

### What is it?
The neuromorphs pipeline is a small python pipeline for the analysis of 2-photon and neuropixels data.
It is primarily designed to analyze the responses of V1 after showing certain stimuli, in this case texture morphs.
This was created by me, Floris Hoogenboom, as part of my internship at the Netherlands institute for Neuroscience. 

### How does it work?
In the config file, point to a folder containing different sessions you want to analyze. Make sure each session you want to analyze has a process.flag in its root.
This program is able to analyze and perform PCA on the image files (the stimuli), neuropixels data, and two-photon data.
In order to determine how the stimuli relate to the neural responses, we use PCA on both the image data and the neural data. 


### Folder structure
-- root_folder

-----> session_folder_1

-----------> 2p_data

-----------> stimuli

-----> session_folder_2

-----------> neuropixels_data

-----------> stimuli

### Debug stuff
Basically, run the run_pipeline.py. This will initiate a Pipeline() object. 
This object gets information from config/settings.json
Then, the script will check in the session folder if there is a 2p, ephys, and stimuli folder. If these are present, it will 
tell the pipeline object to process the data. This is done by using seperate objects, like Neuropixels(), Two_photon(), and Stimuli().
These basically all do the same, but use different data sources.
#### What they do in common
- Load the data
- perform PCA on said data
- return a dictionary called pca_dict. Each entry in this dictionary has three items:
  1. The PCA coordinates of the items, grouped per stimulus. (for example, if all stimuli are shown 5 times in a session, it will have PCA coordinates of the averaged activation of these five repetitions)
  2. The variance explained per component
  3. The labels of each item. (This is a dataframe with all the metadata entries that are selected in the config file)