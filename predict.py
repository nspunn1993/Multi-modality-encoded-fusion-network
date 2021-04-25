import os

from train import config
from unet3d.prediction import run_validation_cases


def main():
    prediction_dir = "prediction/"
    print("Model file: {}".format(config["model_file"]))
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=["t1", "t1ce", "flair", "t2"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    main()
