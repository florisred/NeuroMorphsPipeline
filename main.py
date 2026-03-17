from src.pipeline import Pipeline

if __name__ == "__main__":

    nm = Pipeline()
    nm.load_two_photon()
    nm.load_stimuli()
    nm.create_plots()