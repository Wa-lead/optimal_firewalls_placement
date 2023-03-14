import numpy as np
import matplotlib.pyplot as plt
from GraPart.partition import MultiwayPartioning, OneWayPartitioning, Bisection
from GraPart.environment import Environment
import pandas as pd

def _find_number_of_firewalls(environment, variation):
    if variation == "self":
        return len(environment.find_firewalls(variation='self'))
    else:
        return len(environment.find_firewalls(variation='other')[0])

def single_network_showcase(
    xy: np.ndarray=None,
    num_nodes: int = 400,
    connect_threshold: float= 1,
    max_size: int = 50,
    xMax: float = 20,
    yMax:float = 20,
    variation: str = "self",
    ):
    """
    This function is used to show the partition of a single network
    :param num_nodes: the number of nodes in the network
    :param connect_threshold: the distance of the connection between nodes
    :param max_size: the number of clusters in the network
    :param xMax: the maximum of the x coordinate
    :param yMax: the maximum of the y coordinate
    :return: the figure of the partition
    """
    
    if xy is None:
        x = np.random.uniform(size=num_nodes, low=0, high=xMax)
        y = np.random.uniform(size=num_nodes, low=0, high=yMax)
        xy = np.array([x,y]).T.astype('float32')

        
    number_of_firewalls = [0,0,0]
    # setup the network
    environment = Environment(xy, max_size, connect_threshold)
    fig,_ = environment.plot()
    number_of_firewalls[0] = _find_number_of_firewalls(environment, variation)
        
    # partition the network
    environment = MultiwayPartioning().fit(environment)
    fig2,_ = environment.plot()
    number_of_firewalls[1] = _find_number_of_firewalls(environment, variation)


    # partition the network
    environment = OneWayPartitioning().fit(environment)
    fig3,_ = environment.plot()
    number_of_firewalls[2] = _find_number_of_firewalls(environment, variation)

    return (fig,fig2,fig3), number_of_firewalls



# def bisection_showcase(num_nodes: int = 400,
#     max_size: int = 50,
#     max_firewalls : int= 30,
#     connect_threshold: float = 1,
#     xMax : float= 20,
#     yMax: float= 20,
#     variation: str = "self",
#     max_iter = 100):
#     """
#     This function is used to show the partition of a single network
#     :param num_nodes: the number of nodes in the network
#     :param connect_threshold: the distance of the connection between nodes
#     """
#     x = np.random.uniform(size=num_nodes, low=0, high=xMax)
#     y = np.random.uniform(size=num_nodes, low=0, high=yMax)
#     xy = np.array([x,y]).T.astype('float32')
#     results = []
#     best_run_buffer = []
#     for _ in range(max_iter):
#         if variation == "self":
#             results += (bisection(xy, max_clusters = max_clusters,
#                                     max_firewalls = max_firewalls,
#                                     connect_threshold = connect_threshold,
#                                     variation = "self", margin = margin, return_results=False))
#         elif variation == "other":
#             results+= (bisection(xy, max_clusters = max_clusters,
#                                     max_firewalls = max_firewalls,
#                                     connect_threshold = connect_threshold,
#                                     variation = "other", margin = margin, return_results=False))

#     results = pd.DataFrame(data = results, columns = results[0].keys())
#     results = results.groupby('clusters').mean().reset_index().to_dict('records')
#     filtered_results = [i for i in results if i['firewalls'] < max_firewalls]
#     best_ = max(filtered_results, key = lambda x: x['clusters'])['clusters']
#     return results, single_network_showcase(xy=xy,
#                                             connect_threshold = connect_threshold,
#                                             max_size = best_,
#                                             xMax = xMax,
#                                             yMax = yMax,
#                                             variation = variation, margin=margin)

    

        
        
    
    