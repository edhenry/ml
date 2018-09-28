import numpy as np
import os
import pandas as pd
from imgaug import augmenters as iaa

def get_sample_counts(output_dir: str, datasets: str, class_names: list):
    """
    Class-wise positive sample count of a dataset

    
    Arguments:
        output_dir {str} -- folder containing the dataset.csv file
        datasets {str} -- train|validation|test set(s)
        class_names {list of str} -- target classes 
    """

    df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
    total_count = df.shape[0]
    labels = df[class_names].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts


def get_class_weights(total_counts: int, class_positive_counts: dict, multiply: int):
    """Calculate the class_weight used in training

    Arguments:
        total_counts {int} -- total counts (name implies)
        class_positive_counts {dict} -- dict of int, eg. {"Effusion": 300, "Infiltration": 300}
        multiply {int} -- positive weight multiply
    """

    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }
    
    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weight = []
    for i, class_name in enumerate(class_names):
        class_weight.append(get_single_class_weight(label_counts[i], total_counts))


def augmenter():
    """
    Method to augment images.

    Following from CheXNet paper, images were randomly flipped with 50% probability

    """
    augmenter = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
        ],
        random_order=True
    )
    return augmenter


