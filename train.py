import os
import glob
import time
import numpy as np
from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import aefusionnetwork_model
from unet3d.model import aedilatedfusionnetwork_model
from unet3d.training import load_old_model, train_model
	
def get_whole_tumor_mask(data):
    return data > 0

def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)

def get_enhancing_tumor_mask(data):
    return data == 4

def fetch_training_data_files():
    print("fetch_training_data_files")
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(config["train_path"], "*", "*")):
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files

def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    
    return callbacks

def return_old_model(config):
    model = aefusionnetwork_model(input_shape=config["input_shape"], n_labels=3,
              initial_learning_rate=config["initial_learning_rate"],
              n_base_filters=config["n_base_filters"], multi_gpu_flag=True)
              
    if os.path.exists(config["model_file"]):
        model.load_weights(config["model_file"])
    
    return model

def main(overwrite=True):
    config = dict()

    config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
    config["image_shape"] = (144, 144, 144)  # This determines what shape the images will be cropped/resampled to.
    config["n_base_filters"] = 8
    config["patch_shape"] = None  # switch to None to train on the whole image
    config["labels"] = (1,2,4)  # the label numbers on the input image
    config["n_labels"] = len(config["labels"])
    config["all_modalities"] = ["t1"]
    config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
    config["nb_channels"] = len(config["training_modalities"])
    if "patch_shape" in config and config["patch_shape"] is not None:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
    else:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
    config["truth_channel"] = config["nb_channels"]
    config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

    config["batch_size"] = 2
    config["validation_batch_size"] = 12
    config["n_epochs"] = 500  # cutoff the training after this many epochs
    config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
    config["early_stop"] = 80  # training will be stopped after this many epochs without the validation loss improving
    config["initial_learning_rate"] = 0.00001
    config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
    config["validation_split"] = 0.8  # portion of the data that will be used for training
    config["flip"] = False  # augments the data by randomly flipping an axis during
    config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
    config["distort"] = None  # switch to None if you want no distortion
    config["augment"] = config["flip"] or config["distort"]
    config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
    config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
    config["skip_blank"] = True  # if True, then patches without any target will be skipped
    config["train_path"]="/home/sonali/BDALAB/.sonali/modelmaster/"
    config["data_file"] = config["train_path"]+"brats_data.h5"
    config["model_file"] = config["train_path"]+"tumor_segmentation_model_5.h5"
    config["training_log"] = config["train_path"]+"trainingLog/training_5.log"
    config["training_file"] = config["train_path"]+"training_ids.pkl"
    config["validation_file"] = config["train_path"]+"validation_ids.pkl"
    config["overwrite"] = True  # If True, will previous files. If False, will use previously written files.

    training_files = fetch_training_data_files()
    write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"])
    data_file_opened = open_data_file(config["data_file"])

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        model = aefusionnetwork_model(input_shape=config["input_shape"], n_labels=3,
                                      initial_learning_rate=config["initial_learning_rate"],
                                      n_base_filters=config["n_base_filters"], multi_gpu_flag=True)
                                      
    train_generator, validation_generator = get_training_and_validation_generators(
            data_file_opened,
            batch_size=config["batch_size"],
            data_split=config["validation_split"],
            overwrite=overwrite,
            validation_keys_file=config["validation_file"],
            training_keys_file=config["training_file"],
            n_labels=config["n_labels"],
            labels=config["labels"],
            patch_shape=config["patch_shape"],
            validation_batch_size=config["validation_batch_size"],
            validation_patch_overlap=config["validation_patch_overlap"],
            training_patch_start_offset=config["training_patch_start_offset"],
            permute=config["permute"],
            augment=config["augment"],
            skip_blank=config["skip_blank"],
            augment_flip=config["flip"],
            augment_distortion_factor=config["distort"])

    ip_1 = train_generator[0][:,0:1,:,:,:]
    ip_2 = train_generator[0][:,1:2,:,:,:]
    ip_3 = train_generator[0][:,2:3,:,:,:]
    ip_4 = train_generator[0][:,3:4,:,:,:]
    input = [ip_1,ip_2,ip_3,ip_4]
    out_1 = train_generator[1][:,0:1,:,:,:]
    out_2 = train_generator[1][:,1:2,:,:,:]
    out_3 = train_generator[1][:,2:3,:,:,:]
    output = [out_1,out_2,out_3]

    callbacks = get_callbacks(config["model_file"],
                            initial_learning_rate=config["initial_learning_rate"],
                            learning_rate_drop=config["learning_rate_drop"],
                            learning_rate_patience=config["patience"],
                            early_stopping_patience=config["early_stop"])

    results = model.fit(input, train_generator[1], validation_split=1-config["validation_split"], batch_size=config["batch_size"], epochs=config["n_epochs"], callbacks=callbacks)

if __name__ == "__main__":
    main(overwrite=config["overwrite"])