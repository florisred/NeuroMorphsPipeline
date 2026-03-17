import json
from pathlib import Path
def generate_config(
        config_dir,
        settings_file
):
    if not config_dir.is_dir():
        config_dir.mkdir()

    settings_dict = {

        "DATA_FOLDER": "/path/to/data/folder",
        "PSEUDOPOP_EXPORT": "PSEUDOPOP_EXPORT.h5",
        "PSEUDOPOP_DATA": "X",
        "PSEUDOPOP_LABELS": "y/orientation_deg",
        "num_components_pixel_space": 8,
        "num_components_gabor_space": 3,
        "gabor_params": {
            "wavelengths": [4.2, 7.94, 8.4, 11.84, 16.81, 23.78],
            "orientations": [0, 30, 60, 90, 120, 150],  # Degrees
            "gamma": 0.5,
            "grid_size": (8, 8)
        }

    }
    while True:
        data_folder = input("Please enter the data folder: ")
        try:
            data_folder_test = Path(data_folder)
            break
        except Exception as e:
            print(e)
            print('try again, could not find that path')

    settings_dict["DATA_FOLDER"] = data_folder


    with open(settings_file, 'w') as outfile:
        outfile.write(json.dumps(settings_dict, indent=4))

