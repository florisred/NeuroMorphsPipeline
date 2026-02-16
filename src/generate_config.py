import json

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
    with open(settings_file, 'w') as outfile:
        outfile.write(json.dumps(settings_dict, indent=4))

