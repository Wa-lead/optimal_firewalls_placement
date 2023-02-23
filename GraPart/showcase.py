import numpy as np
import matplotlib.pyplot as plt
from GraPart.firewall import find_firewalls
from GraPart.plot import plot_nodes, plot_edges, plot_firewalls
from GraPart.partition import multiway_partitioning, one_way_partitioing, bisection
from GraPart.setup import setup
import pandas as pd

def single_network_showcase(
    xy=None,
    num_nodes = 300,
    connect_distance = 1,
    num_clusters = 5,
    xMax = 20,
    yMax = 20,
    variation = "self",
    margin = 0.1
    ):
    """
    This function is used to show the partition of a single network
    :param num_nodes: the number of nodes in the network
    :param connect_distance: the distance of the connection between nodes
    :param num_clusters: the number of clusters in the network
    :param xMax: the maximum of the x coordinate
    :param yMax: the maximum of the y coordinate
    :return: the figure of the partition
    """
    if xy is None:
        x = np.random.uniform(size=num_nodes, low=0, high=xMax)
        y = np.random.uniform(size=num_nodes, low=0, high=yMax)
        xy = np.array([x,y]).T.astype('float32')

        
    number_of_firewalls = [0,0,0]
    fig, ax = plt.subplots()
    # setup the network
    connect_matrix, labels_list, group_matrix = setup(xy, num_clusters, connect_distance)
    ax = plot_nodes(xy, labels_list, ax)
    ax = plot_edges(xy, connect_matrix, ax)
    ax = plot_firewalls(xy, labels_list, connect_matrix, variation, ax)
    if variation == "self":
        number_of_firewalls[0] = len(find_firewalls(labels_list, connect_matrix, xy, variation))
    else:
        number_of_firewalls[0] = len(find_firewalls(labels_list, connect_matrix, xy, variation)[0])
        
    fig2, ax2= plt.subplots()
    # partition the network
    group_matrix, labels_list = multiway_partitioning(xy, group_matrix, labels_list, connect_matrix)
    ax2 = plot_nodes(xy, labels_list, ax2)
    ax2 = plot_edges(xy, connect_matrix, ax2)
    ax2 = plot_firewalls(xy, labels_list, connect_matrix, variation, ax2)
    if variation == "self":
        number_of_firewalls[1] = len(find_firewalls(labels_list, connect_matrix, xy, variation))
    else:
        number_of_firewalls[1] = len(find_firewalls(labels_list, connect_matrix, xy, variation)[0])


    fig3, ax3 = plt.subplots()
    # partition the network
    group_matrix, labels_list = one_way_partitioing(xy, group_matrix, labels_list, connect_matrix, margin = margin)
    ax3 = plot_nodes(xy, labels_list, ax3)
    ax3 = plot_edges(xy, connect_matrix, ax3)
    ax3 = plot_firewalls(xy, labels_list, connect_matrix, variation, ax3)
    if variation == "self":
        number_of_firewalls[2] = len(find_firewalls(labels_list, connect_matrix, xy, variation))
    else:
        number_of_firewalls[2] = len(find_firewalls(labels_list, connect_matrix, xy, variation)[0])

    
    return (fig,fig2,fig3), number_of_firewalls



def bisection_showcase(num_nodes = 300,
    max_clusters = 50,
    max_firewalls = 30,
    connect_distance = 1,
    xMax = 20,
    yMax = 20,
    variation = "self",
    margin = 0.1,
    max_iter = 100):
    """
    This function is used to show the partition of a single network
    :param num_nodes: the number of nodes in the network
    :param connect_distance: the distance of the connection between nodes
    """
    x = np.random.uniform(size=num_nodes, low=0, high=xMax)
    y = np.random.uniform(size=num_nodes, low=0, high=yMax)
    xy = np.array([x,y]).T.astype('float32')
    results = []
    best_run_buffer = []
    for _ in range(max_iter):
        if variation == "self":
            results += (bisection(xy, max_clusters = max_clusters,
                                    max_firewalls = max_firewalls,
                                    connect_distance = connect_distance,
                                    variation = "self", margin = margin, return_results=True))
        elif variation == "other":
            results+= (bisection(xy, max_clusters = max_clusters,
                                    max_firewalls = max_firewalls,
                                    connect_distance = connect_distance,
                                    variation = "other", margin = margin, return_results=True))


    results = pd.DataFrame(data = results, columns = results[0].keys())
    results = results.groupby('clusters').mean().reset_index().to_dict('records')
    filtered_results = [i for i in results if i['firewalls'] < max_firewalls]
    best_ = max(filtered_results, key = lambda x: x['clusters'])['clusters']
    return results, single_network_showcase(xy=xy,
                                            connect_distance = connect_distance,
                                            num_clusters = best_,
                                            xMax = xMax,
                                            yMax = yMax,
                                            variation = variation, margin=margin)

    

        
        
    
    