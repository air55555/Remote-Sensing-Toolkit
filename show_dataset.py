# Visualization
import seaborn as sns
import visdom

import os
from utils import display_dataset, explore_spectrums, convert_to_color_, \
    plot_spectrums, convert_from_color_, display_predictions, display_lidar_data

from datasets import *
import argparse


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")

parser.add_argument('--with_exploration', action='store_true',
                    help="See data exploration visualization")
parser.add_argument('--dataset', type=str, default=None, choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                                               "datasets (defaults to the current working directory).",
                    default="./Datasets/")
args = parser.parse_args()

FOLDER = args.folder
DATASET = args.dataset
DATAVIZ = args.with_exploration

viz = visdom.Visdom(env=DATASET)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

# Load the dataset
img, img2, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
print(LABEL_VALUES)
# Number of classes
# N_CLASSES = len(LABEL_VALUES) -  len(IGNORED_LABELS)
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

#  Parameters for the SVM grid search
SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                    'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}

# Labels legend
for i, val in enumerate(LABEL_VALUES):
    rect = np.full((40, 40), i, dtype=int)
    display_predictions(convert_to_color(rect), viz,
                        caption="Label name -- {} --- with number: {}".format(LABEL_VALUES[i], i))

# Show the ground truth
display_predictions(convert_to_color(gt), viz, caption="Ground truth")

display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
display_lidar_data(img2, viz)

color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz,
                                       ignored_labels=IGNORED_LABELS)
    plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')
