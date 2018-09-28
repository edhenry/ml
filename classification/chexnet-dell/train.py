import json
import shutil
import os
import pickle
from callback import MultiClassAUROC, MultiGPUModelCheckpoint
from configparser import ConfigParser
from generator import AugmentedImageSequence
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from models.model import ModelFactory
from utility import get_sample_counts, get_class_weights, augmenter

def main():

    # Instantiate config parser
    # as long as a configuration file is in the local directory of this training code
    # it will be utilized by the training script

    # TODO : Add a README for the configuration file used to configure this training cycle
    config_file = "./sample_config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # set a bunch of default config
    output_directory = cp["DEFAULT"].get("output_directory")
    image_source_directory = cp["DEFAULT"].get("image_source_directory")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    # Class names are passed in as array within the configuration script
    class_names = cp["DEFAULT"].get("class_names").split(",")

    # training configuration
    # See sample_config.ini for explanation of all of the parameters
    use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
    use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
    use_best_weights = cp["TRAIN"].getboolean("use_best_weights")
    epochs = cp["TRAIN"].getint("epochs")
    batch_size = cp["TRAIN"].getint("batch_size")
    initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")
    generator_workers = cp["TRAIN"].getint("generator_workers")
    image_dimension = cp["TRAIN"].getint("image_dimension")
    train_steps = cp["TRAIN"].get("train_steps")
    reduce_learning_rate = cp["TRAIN"].getint("reduce_learning_rate")
    min_learning_rate = cp["TRAIN"].getfloat("min_learning_rate")
    validation_steps = cp["TRAIN"].get("validation_steps")

