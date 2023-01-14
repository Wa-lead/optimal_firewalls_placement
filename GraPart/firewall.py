import numpy as np


def find_firewalls(labels_list, connect_matrix, xy, variation = "self"):
    states = np.zeros(len(xy))
    if variation == "self":
        return find_firewall_self(labels_list, connect_matrix, xy, states)
    elif variation == "other":
        return find_firewall_other(labels_list, connect_matrix, xy, states)
    else:
        raise ValueError("Variation must be either 'self' or 'other'.")


def find_firewall_self(labels_list, connect_matrix, xy, states):
    num_clusters = len(np.unique(labels_list))
    num_nodes = len(labels_list)
    for i in range(num_nodes):
        connections = np.zeros(num_clusters)
        for j in range(num_nodes):
            if connect_matrix[i,j] == 1 and states[j] != 1:
                connections[labels_list[j]] += 1
        connections[labels_list[i]] = 0
        if np.sum(connections) > 0:
            states[i] = 1
    return xy[states==1]


def find_firewall_other(labels_list, connect_matrix, xy, states):
    num_clusters = len(np.unique(labels_list))
    num_nodes = len(labels_list)
    for i in range(num_nodes):
        if states[i] == 1 or states[i] == 2:
            continue
        connections = np.zeros(num_clusters)
        for j in range(num_nodes):
            if connect_matrix[i,j] == 1 and states[j] != 1 and states[j] != 2:
                connections[labels_list[j]] += 1
        connections[labels_list[i]] = 0
        if np.sum(connections) > 0:
            states[connect_matrix[i,:] == 1] = 2
            states[i] = 1

    return xy[states == 1], xy[states == 2]

