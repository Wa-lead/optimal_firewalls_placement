
import matplotlib.pyplot as plt
import numpy as np  
from GraPart.firewall import find_firewalls

def plot_nodes(xy, labels_list, ax=None):
    """
    This function is used to plot the nodes
    :param xy: the coordinates of the nodes: (num_nodes,2) matrix
    :param labels_list: the list of the labels of the nodes: (num_nodes) list
    :return: the figure of the nodes
    """
    # plot the nodes
    ax.scatter(xy[:, 0], xy[:, 1], c=labels_list)
    return ax


def plot_edges(xy, connect_matrix, ax=None):
    """
    This function is used to plot the edges
    :param xy: the coordinates of the nodes: (num_nodes,2) matrix
    :param connect_matrix: the matrix of the connections between nodes: (num_nodes, num_nodes) matrix
    :return: the figure of the edges
    """
    # plot the edges
    for i in range(len(xy)):
        for j in range(len(xy)):
            if connect_matrix[i, j] > 0:
                ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], 'k-')
    return ax


def plot_firewalls(xy,labels_list, connect_matrix, variation = "self", ax=None, color = "green", color2='orange'):
    """
    This function is used to plot the firewalls
    :param xy: the coordinates of the nodes: (num_nodes,2) matrix
    :param firewall: the coordinates of the firewalls: (num_firewalls,2) matrix
    :return: the figure of the firewalls
    """
    if variation == "self":
        firewall = find_firewalls(labels_list, connect_matrix, xy, variation)
        ax.scatter(firewall[:, 0], firewall[:, 1], c=color)
        return ax
    elif variation == "other":
        firewall1, protected = find_firewalls(labels_list, connect_matrix, xy, variation)
        ax.scatter(firewall1[:, 0], firewall1[:, 1], c=color)
        ax.scatter(protected[:, 0], protected[:, 1], c=color2)
        return ax