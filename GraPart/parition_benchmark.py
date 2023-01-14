"""
This is exactly the same as the regular parition, but this also returns the number of iterations.
"""


import time
import numpy as np
from GraPart.firewall import find_firewalls
from GraPart.setup import setup
import matplotlib.pyplot as plt


def multiway_partitioning(xy, group_matrix, labels_list, connect_matrix):
    labels_list = labels_list.copy()
    num_clusters = group_matrix.shape[0]
    cluster_pairs = [(i, j) for i in range(num_clusters)
                     for j in range(i+1, num_clusters) if group_matrix[i, j] == 1]
    count = 0
    for i, j in cluster_pairs:
        count += 1
        i, j = int(i), int(j)
        group_matrix, labels_list, INTERCHANGE = two_way_partitioning(
            i, j, xy, group_matrix, labels_list, connect_matrix)
        if INTERCHANGE:
            for idx in range(len(group_matrix[i])):
                if idx != j and idx != i and group_matrix[i, idx] == 1:
                    cluster_pairs.append((i, idx))
            for idx in range(len(group_matrix[j])):
                if idx != i and idx != j and group_matrix[j, idx] == 1:
                    cluster_pairs.append((j, idx))

    return group_matrix, labels_list, count


def one_way_partitioing(xy, group_matrix, labels_list, connect_matrix, margin=0.1):
    num_clusters = group_matrix.shape[0]
    pairs = [(i, j) for i in range(num_clusters)
             for j in range(num_clusters) if group_matrix[i, j] == 1]
    count = 0
    while pairs:
        i, j = pairs.pop()
        count += 1
        group_matrix, labels_list, INTERCHANGE = one_way_partitioing_single(
            i, j, xy, group_matrix, labels_list, connect_matrix, original_k=num_clusters, margin=margin)
        if INTERCHANGE:
            # group j ate all the nodes in group i
            if num_clusters > group_matrix.shape[0]:
                pairs = [(i, j) for i in range(group_matrix.shape[0]) for j in range(
                    i+1, group_matrix.shape[0]) if group_matrix[i, j] == 1]
            else:
                for idx in range(len(group_matrix[i])):
                    if idx != i and group_matrix[i, idx] == 1:
                        pairs.append((i, idx))
                        pairs.append((idx, i))
                for idx in range(len(group_matrix[j])):
                    if idx != j and idx != i and group_matrix[j, idx] == 1:
                        pairs.append((j, idx))
                        pairs.append((idx, j))

    return group_matrix, labels_list, count


def two_way_partitioning(group1, group2, xy, group_matrix, labels_list, connect_matrix):
    """
    This function is used to partition the graph into two groups
    :param group1: the first group: int
    :param group2: the second group: int
    :param group_matrix: the matrix of the groups: kxk matrix
    :param xy: the coordinates of the nodes: (num_nodes,2) matrix
    :param connect_matrix: the matrix of the connections between nodes: (num_nodes, num_nodes) matrix
    :param dvalue_list: the matrix of the dvalues of the nodes: (num_nodes) list
    :return: if an INTERCHANGE happened or not
    """
    PENDING = 10000000
    INTERCHANGE = False
    # check if the two groups are connected
    if group_matrix[group1, group2] == 0:
        return group_matrix, labels_list, False
    else:
         # Extract the indices of the nodes in each group
        group1_nodes = np.where(labels_list == group1)[0]
        group2_nodes = np.where(labels_list == group2)[0]

        # Initialize the dvalue_list with zeros
        dvalue_list = np.zeros(len(xy))

        # Calculate the dvalue for the nodes in group1
        dvalue_list[group1_nodes] = np.sum(connect_matrix[group1_nodes][:, group2_nodes], axis=1) - np.sum(connect_matrix[group1_nodes][:, group1_nodes], axis=1)

        # Calculate the dvalue for the nodes in group2
        dvalue_list[group2_nodes] = np.sum(connect_matrix[group2_nodes][:, group1_nodes], axis=1) - np.sum(connect_matrix[group2_nodes][:, group2_nodes], axis=1)

        while True:
            total_gain = 0
            buffer = []
            labels_list_copy = labels_list.copy()

            x_star = []
            y_star = []

            # find the group of minimum size
            threshold = min(len(group1_nodes), len(group2_nodes))
            if threshold == 0 or group_matrix[group1, group2] == 0:
                break

            # exhaust every point INTERCHANGE
            for _ in range(threshold):
                # find the max gain
                group1_nodes, group2_nodes = np.meshgrid(group1_nodes, group2_nodes)
                # Calculate the gain for each combination of nodes
                gains = dvalue_list[group1_nodes] + dvalue_list[group2_nodes] - 2 * connect_matrix[group1_nodes, group2_nodes]
                # Get the maximum gain and the indices of the nodes that give the maximum gain
                node1, node2 = np.unravel_index(np.argmax(gains), gains.shape)
                gain = gains[node1, node2]
                node1, node2 = group1_nodes[node1, node2], group2_nodes[node1, node2]
                
                x_star.append(node1)
                y_star.append(node2)
                # remove the nodes from consideration for the next iteration
                labels_list_copy[node1] = PENDING
                labels_list_copy[node2] = PENDING

                # Extract the indices of the nodes in each group
                group1_nodes = np.where(labels_list_copy == group1)[0]
                group2_nodes = np.where(labels_list_copy == group2)[0]

                # Initialize the dvalue_list with zeros
                dvalue_list = np.zeros(len(xy))

                # Calculate the dvalue for the nodes in group1
                dvalue_list[group1_nodes] = np.sum(connect_matrix[group1_nodes][:, group2_nodes], axis=1) - np.sum(connect_matrix[group1_nodes][:, group1_nodes], axis=1)

                # Calculate the dvalue for the nodes in group2
                dvalue_list[group2_nodes] = np.sum(connect_matrix[group2_nodes][:, group1_nodes], axis=1) - np.sum(connect_matrix[group2_nodes][:, group2_nodes], axis=1)

                # update the total gain
                total_gain += gain
                buffer.append({'total_gain': total_gain, 'x_star': x_star.copy(), 'y_star': y_star.copy()})

            max_total_gain = max(buffer, key=lambda x: x['total_gain'])
            if max_total_gain['total_gain'] > 0:
                INTERCHANGE = True
                labels_list[max_total_gain['x_star']] = group2
                labels_list[max_total_gain['y_star']] = group1
            else:
                 break

            k = group_matrix.shape[0]
            group_matrix = np.zeros((k, k))
            for i in range(k):
                for j in range(i + 1, k):
                    if np.sum(connect_matrix[labels_list == i][:, labels_list == j]) > 0:
                        group_matrix[i, j] = 1
                        group_matrix[j, i] = 1
                        
        return group_matrix, labels_list, INTERCHANGE

def one_way_partitioing_single(group1, group2, xy, group_matrix, labels_list, connect_matrix, original_k, margin=0.1):
    """
    Performs a one way partitioning on two groups, group2 takes nodes form group1 given a heuristic.
    Hueristic: select the point with the highest Dvalue and the smallest cluster size.

    :param group1: The index of the first group
    :param group2: The index of the second group
    :param xy: The data points
    :param group_matrix: The matrix that keeps track of the connections between the groups
    :param labels_list: The list that keeps track of the group of each node
    :param connect_matrix: The matrix that keeps track of the connections between the nodes
    :param original_k: The original number of groups: to track the original size because group matrix keeps changing.
    :param margin: The margin of error for the size of the groups
    """



    # If the two clustes are not connected, return
    if group_matrix[group1, group2] == 0:
        return group_matrix, labels_list, False

    else:

        # Performs depth first search to find the set of nodes forming the graph
        def dfs(node, connect_matrix, labels_list):
            """
            Performs depth first search to find the set of nodes forming the graph

            :param node: The node to start from
            :param connect_matrix: The matrix that keeps track of the connections between the nodes
            :param labels_list: The list that keeps track of the group of each node
            :return: The set of nodes forming the graph
            """
            visited = set()
            stack = [node]
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    # Get the indices of the nodes connected to the current vertex
                    # and add them to the stack if they haven't been visited yet
                    connected_nodes = set(np.where(connect_matrix[vertex] == 1)[0])
                    connected_nodes = set([node for node in connected_nodes if labels_list[node] == labels_list[vertex]])
                    stack.extend(connected_nodes - visited)
            return list(visited)


        # Flag to check if the partitioning was successful
        INTERCHANGE = False
        # The upper bound for the size of the group, original_k because the group_matrix keeps changing and we want to keep track of the original size.
        DEFAULT_SIZE = len(xy) / original_k
        # The limit in which a cluster whould be within
        UPPER_BOUND = (1+margin) * DEFAULT_SIZE

    
        # Extract the indices of the nodes in each group
        group1_nodes = np.where(labels_list == group1)[0]
        group2_nodes = np.where(labels_list == group2)[0]

        # only nodes from group1_nodes that are connected to group2_nodes are considered
        group1_edge_nodes = [i for i in group1_nodes if np.sum(connect_matrix[i, labels_list == group2]) > 0]

        # Set the threshold for the loop
        threshold = int(UPPER_BOUND - len(group2_nodes))

        # Compute dvalues
        dvalue_list = np.zeros(len(xy))
        for i in group1_edge_nodes:
            dvalue_list[i] = np.sum(connect_matrix[i, labels_list == group2]) - np.sum(connect_matrix[i, labels_list == group1])

        # Initialize the labels_list_copy that will be used in the loop
        labels_list_copy = labels_list.copy()

        # Loop until the upper bound is reached, or both groups are no longer connected
        while len(group2_nodes) < UPPER_BOUND  and group_matrix[group1, group2] == 1:
            # ئreate a list that contains every node conntected node from the same group
            group1_connected_nodes = []

            for i in group1_edge_nodes:
                group = dfs(i, connect_matrix, labels_list_copy)
                group1_connected_nodes.append((i, group))

            # Select the node with highest dvalue and smallest conntect group.
            # Reasoning: largest size will guarntee removing a single firewall, bunch of small ones could remove more.
            group1_connected_nodes = [i for i in group1_connected_nodes if len(i[1]) <=  threshold]
            selected_group = max(group1_connected_nodes, key=lambda x: (-1*len(x[1]), dvalue_list[x[0]]))[1] if group1_connected_nodes else None

            # If there is a valid group, interchange the labels
            if selected_group:
                labels_list_copy[selected_group] = group2
                group1_nodes = np.where(labels_list_copy == group1)[0]
                group2_nodes = np.where(labels_list_copy == group2)[0]

                # Update the threshold after the interchange
                threshold = int(UPPER_BOUND - len(group2_nodes))

                # only nodes from group1_nodes that are connected to group2_nodes are considered
                group1_edge_nodes = [i for i in group1_nodes if np.sum(connect_matrix[i, labels_list_copy == group2]) > 0]
                
                
                # Incase group2 ate all the nodes from group1
                # Find all the group numbers larger than group1 and decrease them by 1, to maintain consecutive numbering
                if len(group1_nodes) == 0:
                    # Find the group numbers larger than group1
                    group_numbers = np.arange(group1+1, group_matrix.shape[0])
                    # Decrease the group numbers by 1
                    labels_list_copy[np.isin(labels_list_copy, group_numbers)] -= 1
                    
                    # update the group matrix
                    groups_count = len(np.unique(labels_list_copy))
                    group_matrix = np.zeros((groups_count,groups_count))
                    for i in range(groups_count):
                        for j in range(i + 1, groups_count):
                            if np.sum(connect_matrix[labels_list_copy == i][:, labels_list_copy == j]) > 0:
                                group_matrix[i, j] = 1
                                group_matrix[j, i] = 1
                    INTERCHANGE = True
                    break

                
                # Update the group matrix
                groups_count = len(np.unique(labels_list_copy))
                group_matrix = np.zeros((groups_count, groups_count))
                for i in range(groups_count):
                    for j in range(i + 1, groups_count):
                        if np.sum(connect_matrix[labels_list_copy == i][:, labels_list_copy == j]) > 0:
                            group_matrix[i, j] = 1
                            group_matrix[j, i] = 1

                # Compute dvalues for the next iteration
                dvalue_list = np.zeros(len(xy))
                for i in group1_nodes:
                    dvalue_list[i] = np.sum(connect_matrix[i, labels_list_copy == group2]) - np.sum(connect_matrix[i, labels_list_copy == group1])

                
                INTERCHANGE = True
            else:
                break
            
        return group_matrix, labels_list_copy, INTERCHANGE    

def bisection(xy,
            max_clusters = 30,
            connect_distance = 1,
            max_firewalls = 50, 
            variation='self', margin=0.1):

    possible_clusters = [i for i in range(2,max_clusters+1)]
    buffer = []

    # start bisecting
    high = len(possible_clusters)-1
    low = 0

    while high - low > 1:
        mid = (high + low) // 2
        connect_matrix, labels_list, group_matrix = setup(xy, possible_clusters[mid], connect_distance)
        group_matrix, labels_list, _ = multiway_partitioning(xy, group_matrix, labels_list, connect_matrix)
        group_matrix, labels_list, _ = one_way_partitioing(xy, group_matrix, labels_list, connect_matrix, margin = margin)
        if variation == 'self':
            firewall_count = len(find_firewalls(labels_list, connect_matrix, xy, variation = "self"))
        elif variation == 'other':
            firewall_count = len(find_firewalls(labels_list, connect_matrix, xy, variation = "other")[0])
        
        buffer.append({'max_firewalls': max_firewalls, 'clusters': possible_clusters[mid], 'firewalls': firewall_count})
        if firewall_count > max_firewalls:
            high = mid 
        else:
            low = mid 
    
    return buffer


if __name__ == "__main__":

    num_nodes = 500
    connect_distance = 1
    num_clusters = 5
    xMin = 0
    xMax = 20
    yMin = 0
    yMax = 20
    x = np.random.uniform(size=num_nodes, low=xMin, high=xMax)
    y = np.random.uniform(size=num_nodes, low=yMin, high=yMax)
    xy = np.array([x, y]).T.astype('float32')

    bisection(xy, max_clusters=30, connect_distance=1,
              max_firewalls=50, variation='self')
