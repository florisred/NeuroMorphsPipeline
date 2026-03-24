from src.pipeline import Pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":

    nm = Pipeline()
    nm.load_two_photon(split=False)
    nm.load_stimuli()
    nm.create_plots()