# Forked from https://github.com/yolish/transposenet
import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath
import time
from os import mkdir, getcwd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import itertools

# Logging and output utils
##########################
def get_stamp_from_log():
    """
    Get the time stamp from the log file
    :return:
    """
    return split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log","")


def create_output_dir(name):
    """
    Create a new directory for outputs, if it does not already exist
    :param name: (str) the name of the directory
    :return: the path to the outpur directory
    """
    out_dir = join(getcwd(), name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir


def init_logger():
    """
    Initialize the logger and create a time stamp for the file
    """
    path = split(realpath(__file__))[0]

    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        filename = log_config_dict.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%d_%m_%y_%H_%M", time.localtime()), ".log"])

        # Creating logs' folder is needed
        log_path = create_output_dir('out')

        log_config_dict.get('handlers').get('file_handler')['filename'] = join(log_path, filename)
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)

# Plotting utils
##########################
def plot_loss_func(sample_count, loss_vals, loss_fig_path):
    plt.figure()
    plt.plot(sample_count, loss_vals)
    plt.grid()
    plt.title('Loss')
    plt.xlabel('Number of samples')
    plt.ylabel('Loss')
    plt.savefig(loss_fig_path)


def plot_confusion_matrix(cm,
                          target_names,
                          font_size=14,
                          out_dir = None,
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a CM plot
    :param cm: confusion matrix from sklearn.metrics.confusion_matrix
    :param target_names: given classification classes such as ['high', 'medium', 'low']
    :param font_size: (int) CM font size of CM text
    :param out_dir: (str) dir to plot the CM
    :param cmap: the gradient of the values displayed from matplotlib.pyplot.cm
                 plt.get_cmap('jet') or plt.cm.Blues
    :param normalize: If False, plot the raw numbers, If True, plot the proportions
    Citiation: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        #cmap = plt.get_cmap('Pastel1')

    fig = plt.figure()
    fig.set_size_inches(32, 18)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    clb = plt.colorbar()
    #clb.set_label('Number of samples')
    clb.ax.set_title('# of samples', size=font_size)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, size=font_size)  #, rotation=45
        plt.yticks(tick_marks, target_names, size=font_size)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.title('CM, Normalized - Accuracy={:0.4f}; Misclass={:0.4f}'.format(accuracy, misclass), size=font_size)
    else:
        plt.title('CM - Accuracy={:0.4f}; Misclass={:0.4f}'.format(accuracy, misclass), size=font_size)

    import matplotlib.font_manager as font_manager
    font_prop = font_manager.FontProperties(size=font_size)
    thresh = np.nanmax(cm) / 1.5 if normalize else np.nanmax(cm) / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:2.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontproperties=font_prop)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=font_size)
    plt.xlabel('\nPredicted label', size=font_size)
    plt.savefig(join(out_dir + '_CM.png'), bbox_inches='tight', dpi=100)
    #plt.show()


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x