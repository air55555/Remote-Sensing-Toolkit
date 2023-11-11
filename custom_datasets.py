from utils import *
import os
import warnings
from numpy import loadtxt
warnings.filterwarnings('ignore')





CUSTOM_DATASETS_CONFIG = {

    '10000_exp': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },
    '10000_bint': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },
    '10000_plast': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },
    '10000_plast_white': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },
    'no_lamp': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },
    'M_10000_bint': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },'M_10000_cart': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },'M_10000_exp': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },'M_10000_plast': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },'M_10000_plast_white': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },

    'M_20000_cart_white': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    },
    'buckwheat': {
'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'image.hdr',
        'gt': 'array_1.npz',
        'download': False,
        'loader': lambda folder: exp_loader(folder)
    }


}
def get_good_indices(name=None):
    """
    Returns indices of bands which are not noisy

    Parameters:
    ---------------------
    name: name
    Returns:
    -----------------------
    numpy array of good indices
    """
    indices = np.arange(272)
    indices = indices[70:120]
    return indices
def  exp_loader(folder,noGT = False):
    palette = None
    img = my_open_file(folder + 'tuy.hdr')  # [:, :, :-2]

    #img = img[:, :, get_good_indices()]
    import matplotlib.pyplot as plt
    import numpy as np

    def show_csv_as_image(csv_file):
        # Load CSV data into a 2D NumPy array
        data = np.loadtxt(csv_file, delimiter=',')

        # Display the 2D array as an image
        plt.imshow(data, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()

    # Example usage
    csv_file_path = folder+'small.csv'
    # show_csv_as_image(csv_file_path)

   # removal of damaged sensor line
    # img = np.delete(img, 445, 0)
    # img = img[:, :, get_good_indices()]  # [:, :, get_good_indices(name)]

    # gt = gt.astype('uint8')

    rgb_bands = (7, 13, 15)
    if not noGT:
        gt = my_open_file(folder + 'array_1.npz')
        #gt = gt[:, get_good_indices()]
        label_values = txt_to_lst(folder + "classes.txt")
        label_values =  label_values[:-1]
    # label_values = ["background",
    #                 'tomato',
    #                 'beet',
    #                 'ketchup',
    #                 'tomato juice']
        ignored_labels = [0]
    else:
        gt=None
        label_values=None
        ignored_labels=None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

sett = {'urls': 'https://doi.org/10.5281/zenodo.3984905', 'img': 'image.hdr', 'gt': 'array_1.npz', 'download': False,
        'loader': lambda folder, noGT : exp_loader(folder,noGT)}

CUSTOM_DATASETS_CONFIG = {f'{e}': sett for e in os.listdir(os.getcwd() + '\Datasets')}

print(CUSTOM_DATASETS_CONFIG)

