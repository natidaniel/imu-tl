from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def plot_tsne(X, y, dataset, tsne_fig_path, dim=2, perplexity=30.0, scale_data=False):
    """Perform t-SNE and plot."""
    if dim < 2 or dim > 3:
        raise SystemError("2 <= dim <= 3")
    fig = plt.figure()
    fig.set_size_inches(32, 18)
    if dim == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("2D t-SNE AR features", size=22)
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d', size=22)
        ax.set_title("3D t-SNE AR features")
    if type(X).__module__ == 'torch':
        X = X.cpu().detach().numpy()
    if type(y).__module__ == 'torch':
        y = y.cpu().detach().numpy()
    t_sne = TSNE(n_components=dim, perplexity=perplexity)
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    X_embedded = t_sne.fit_transform(X)
    num_labels = len(np.unique(y))
    labels_names = []
    for l in range(num_labels):
        labels_names.append(dataset.label_dict.get(str(dataset.sorted_unique_raw_labels[l])))
    if dim == 2:
        if num_labels == 2:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
        elif num_labels == 3:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], color='g', marker='v', alpha=0.5, label=labels_names[2])
        elif num_labels == 4:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], color='c', marker='+', alpha=0.5, label=labels_names[3])
        elif num_labels == 5:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], color='m', marker='X', alpha=0.5, label=labels_names[4])
        elif num_labels == 6:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], color='y', marker='<', alpha=0.5, label=labels_names[5])
        elif num_labels == 7:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], color='k', marker='>', alpha=0.5, label=labels_names[6])
        elif num_labels == 8:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], color='k', marker='>', alpha=0.5, label=labels_names[6])
            ax.scatter(X_embedded[y == 7, 0], X_embedded[y == 7, 1], color='orange', marker='*', alpha=0.5, label=labels_names[7])
        elif num_labels == 9:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], color='k', marker='>', alpha=0.5, label=labels_names[6])
            ax.scatter(X_embedded[y == 7, 0], X_embedded[y == 7, 1], color='orange', marker='*', alpha=0.5, label=labels_names[7])
            ax.scatter(X_embedded[y == 8, 0], X_embedded[y == 8, 1], color='darkgreen', marker='1', alpha=0.5, label=labels_names[8])
        elif num_labels == 10:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], color='k', marker='>', alpha=0.5, label=labels_names[6])
            ax.scatter(X_embedded[y == 7, 0], X_embedded[y == 7, 1], color='orange', marker='*', alpha=0.5, label=labels_names[7])
            ax.scatter(X_embedded[y == 8, 0], X_embedded[y == 8, 1], color='darkgreen', marker='1', alpha=0.5, label=labels_names[8])
            ax.scatter(X_embedded[y == 9, 0], X_embedded[y == 9, 1], color='wheat', marker='2', alpha=0.5, label=labels_names[9])
        elif num_labels == 11:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], color='k', marker='>', alpha=0.5, label=labels_names[6])
            ax.scatter(X_embedded[y == 7, 0], X_embedded[y == 7, 1], color='orange', marker='*', alpha=0.5, label=labels_names[7])
            ax.scatter(X_embedded[y == 8, 0], X_embedded[y == 8, 1], color='darkgreen', marker='1', alpha=0.5, label=labels_names[8])
            ax.scatter(X_embedded[y == 9, 0], X_embedded[y == 9, 1], color='wheat', marker='2', alpha=0.5, label=labels_names[9])
            ax.scatter(X_embedded[y == 10, 0], X_embedded[y == 10, 1], color='peru', marker='3', alpha=0.5, label=labels_names[10])
        elif num_labels == 12:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], color='k', marker='>', alpha=0.5, label=labels_names[6])
            ax.scatter(X_embedded[y == 7, 0], X_embedded[y == 7, 1], color='orange', marker='*', alpha=0.5, label=labels_names[7])
            ax.scatter(X_embedded[y == 8, 0], X_embedded[y == 8, 1], color='darkgreen', marker='1', alpha=0.5, label=labels_names[8])
            ax.scatter(X_embedded[y == 9, 0], X_embedded[y == 9, 1], color='wheat', marker='2', alpha=0.5, label=labels_names[9])
            ax.scatter(X_embedded[y == 10, 0], X_embedded[y == 10, 1], color='peru', marker='3', alpha=0.5, label=labels_names[10])
            ax.scatter(X_embedded[y == 11, 0], X_embedded[y == 11, 1], color='slategray', marker='4', alpha=0.5, label=labels_names[11])
        else:
            raise SystemError("2 <= y <= 12")
    else:
        if num_labels == 2:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
        elif num_labels == 3:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], X_embedded[y == 2, 2], color='g', marker='v', alpha=0.5, label=labels_names[2])
        elif num_labels == 4:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], X_embedded[y == 2, 2], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], X_embedded[y == 3, 2], color='c', marker='+', alpha=0.5, label=labels_names[3])
        elif num_labels == 5:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], X_embedded[y == 2, 2], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], X_embedded[y == 3, 2], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], X_embedded[y == 4, 2], color='m', marker='X', alpha=0.5, label=labels_names[4])
        elif num_labels == 6:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], X_embedded[y == 2, 2], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], X_embedded[y == 3, 2], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], X_embedded[y == 4, 2], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], X_embedded[y == 5, 2], color='y', marker='<', alpha=0.5, label=labels_names[5])
        elif num_labels == 7:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], X_embedded[y == 2, 2], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], X_embedded[y == 3, 2], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], X_embedded[y == 4, 2], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], X_embedded[y == 5, 2], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], X_embedded[y == 6, 2], color='k', marker='>', alpha=0.5, label=labels_names[6])
        elif num_labels == 8:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], X_embedded[y == 2, 2], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], X_embedded[y == 3, 2], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], X_embedded[y == 4, 2], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], X_embedded[y == 5, 2], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], X_embedded[y == 6, 2], color='k', marker='>', alpha=0.5, label=labels_names[6])
            ax.scatter(X_embedded[y == 7, 0], X_embedded[y == 7, 1], X_embedded[y == 7, 2], color='orange', marker='*', alpha=0.5, label=labels_names[7])
        elif num_labels == 9:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], X_embedded[y == 2, 2], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], X_embedded[y == 3, 2], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], X_embedded[y == 4, 2], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], X_embedded[y == 5, 2], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], X_embedded[y == 6, 2], color='k', marker='>', alpha=0.5, label=labels_names[6])
            ax.scatter(X_embedded[y == 7, 0], X_embedded[y == 7, 1], X_embedded[y == 7, 2], color='orange', marker='*', alpha=0.5, label=labels_names[7])
            ax.scatter(X_embedded[y == 8, 0], X_embedded[y == 8, 1], X_embedded[y == 8, 2], color='darkgreen', marker='1', alpha=0.5, label=labels_names[8])
        elif num_labels == 10:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], X_embedded[y == 2, 2], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], X_embedded[y == 3, 2], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], X_embedded[y == 4, 2], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], X_embedded[y == 5, 2], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], X_embedded[y == 6, 2], color='k', marker='>', alpha=0.5, label=labels_names[6])
            ax.scatter(X_embedded[y == 7, 0], X_embedded[y == 7, 1], X_embedded[y == 7, 2], color='orange', marker='*', alpha=0.5, label=labels_names[7])
            ax.scatter(X_embedded[y == 8, 0], X_embedded[y == 8, 1], X_embedded[y == 8, 2], color='darkgreen', marker='1', alpha=0.5, label=labels_names[8])
            ax.scatter(X_embedded[y == 9, 0], X_embedded[y == 9, 1], X_embedded[y == 9, 2], color='wheat', marker='2', alpha=0.5, label=labels_names[9])
        elif num_labels == 11:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], X_embedded[y == 2, 2], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], X_embedded[y == 3, 2], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], X_embedded[y == 4, 2], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], X_embedded[y == 5, 2], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], X_embedded[y == 6, 2], color='k', marker='>', alpha=0.5, label=labels_names[6])
            ax.scatter(X_embedded[y == 7, 0], X_embedded[y == 7, 1], X_embedded[y == 7, 2], color='orange', marker='*', alpha=0.5, label=labels_names[7])
            ax.scatter(X_embedded[y == 8, 0], X_embedded[y == 8, 1], X_embedded[y == 8, 2], color='darkgreen', marker='1', alpha=0.5, label=labels_names[8])
            ax.scatter(X_embedded[y == 9, 0], X_embedded[y == 9, 1], X_embedded[y == 9, 2], color='wheat', marker='2', alpha=0.5, label=labels_names[9])
            ax.scatter(X_embedded[y == 10, 0], X_embedded[y == 10, 1], X_embedded[y == 10, 2], color='peru', marker='3', alpha=0.5, label=labels_names[10])
        elif num_labels == 12:
            ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='^', alpha=0.5, label=labels_names[0])
            ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='o', alpha=0.5, label=labels_names[1])
            ax.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1], X_embedded[y == 2, 2], color='g', marker='v', alpha=0.5, label=labels_names[2])
            ax.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1], X_embedded[y == 3, 2], color='c', marker='+', alpha=0.5, label=labels_names[3])
            ax.scatter(X_embedded[y == 4, 0], X_embedded[y == 4, 1], X_embedded[y == 4, 2], color='m', marker='X', alpha=0.5, label=labels_names[4])
            ax.scatter(X_embedded[y == 5, 0], X_embedded[y == 5, 1], X_embedded[y == 5, 2], color='y', marker='<', alpha=0.5, label=labels_names[5])
            ax.scatter(X_embedded[y == 6, 0], X_embedded[y == 6, 1], X_embedded[y == 6, 2], color='k', marker='>', alpha=0.5, label=labels_names[6])
            ax.scatter(X_embedded[y == 7, 0], X_embedded[y == 7, 1], X_embedded[y == 7, 2], color='orange', marker='*', alpha=0.5, label=labels_names[7])
            ax.scatter(X_embedded[y == 8, 0], X_embedded[y == 8, 1], X_embedded[y == 8, 2], color='darkgreen', marker='1', alpha=0.5, label=labels_names[8])
            ax.scatter(X_embedded[y == 9, 0], X_embedded[y == 9, 1], X_embedded[y == 9, 2], color='wheat', marker='2', alpha=0.5, label=labels_names[9])
            ax.scatter(X_embedded[y == 10, 0], X_embedded[y == 10, 1], X_embedded[y == 10, 2], color='peru', marker='3', alpha=0.5, label=labels_names[10])
            ax.scatter(X_embedded[y == 11, 0], X_embedded[y == 11, 1], X_embedded[y == 11, 2], color='slategray', marker='4', alpha=0.5, label=labels_names[11])
        else:
            raise SystemError("2 <= y <= 12")
    # common
    ax.grid()
    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=18)
    #plt.show()
    plt.savefig(tsne_fig_path)
    plt.close()