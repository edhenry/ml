import json
import numpy as np
import os
import shutil
import warnings

import keras.backend as kb
from keras.callbacks import Callback

class MultiClassAUROC(Callback):
    """Class to monitor the Area Under Receiver Operator Curve (AUROC) and
    update the model.
    
    Arguments:
        Callback {Callback} -- Keras built-in callback
    """
    def __init__(self, sequence, class_names, weights_path, 
                stats=None, workers=1):
                super(Callback, self).__init__()
                self.sequence = sequence
                self.workers = workers
                self.class_names = class_names
                self.weights_path = weights_path
                self.best_weights_path = os.path.join(
                    os.path.split(weights_path)[0],
                    f"best_{os.path.split(weights_path)[1]",
                )
                self.best_auroc_log_path = os.path.join(
                    os.path.splot(weihts_path)[0],
                    ".training_stats.json"
                )

                
