import numpy as np
from GraPart.environment import Environment
from copy import deepcopy


def _find_group_matrix(environment: Environment) -> np.ndarray:
    """
    This function finds the group matrix
    :param environment: the environment
    :return: the group matrix
    """
    clusters = np.unique(environment.labels_list)
    group_matrix = np.zeros((len(clusters), len(clusters)))
    for cluster1 in range(len(clusters)):
        cluster1_nodes_idx = np.where(environment.labels_list == clusters[cluster1])[0]
        for cluster2 in range(cluster1+1, len(clusters)):
            cluster2_nodes_idx = np.where(environment.labels_list == clusters[cluster2])[0]
            if np.sum(environment.edge_matrix[cluster1_nodes_idx][:, cluster2_nodes_idx]) > 0:
                group_matrix[cluster1][cluster2] = 1
                group_matrix[cluster2][cluster1] = 1
    return group_matrix
    

class MultiwayPartioning:

    def fit(self, environment: Environment) -> Environment:
        """
        This function performs the multiway partitioning on every pair of clusters in the environment
        :param environment: the environment after the multiway partitioning
        :return: the environment"""

        num_clusters = environment.group_matrix.shape[0]
        cluster_pairs = [(i, j) for i in range(num_clusters) for j in range(
            i+1, num_clusters) if environment.group_matrix[i, j] == 1]
        for i, j in cluster_pairs:
            i, j = int(i), int(j)
            environment, INTERCHANGE = self._two_way_partitioning(
                i, j, environment)
            if INTERCHANGE:
                for idx in range(len(environment.group_matrix[i])):
                    if idx != j and idx != i and environment.group_matrix[i, idx] == 1:
                        cluster_pairs.append((i, idx))
                for idx in range(len(environment.group_matrix[j])):
                    if idx != i and idx != j and environment.group_matrix[j, idx] == 1:
                        cluster_pairs.append((j, idx))
        return environment

    def _two_way_partitioning(self, group1: int, group2: int, environment: Environment):
        """
        This function performs the two way partitioning on the two clusters
        :param group1: the first cluster
        :param group2: the second cluster
        :param environment: the environment
        :return: the environment after the the two way partitioning
        """
        EXCLUDE = 10000000
        INTERCHANGE = False
        # check if the two groups are connected
        if environment.group_matrix[group1, group2] == 0:
            return environment, INTERCHANGE
        else:
            # Extract the indices of the nodes in each group
            group1_nodes = np.where(environment.labels_list == group1)[0]
            group2_nodes = np.where(environment.labels_list == group2)[0]

            # Initialize the dvalue_list with zeros
            dvalue_list = np.zeros(len(environment.xy))

            # Calculate the dvalue for the nodes in group1
            dvalue_list[group1_nodes] = np.sum(environment.edge_matrix[group1_nodes][:, group2_nodes], axis=1) - np.sum(
                environment.edge_matrix[group1_nodes][:, group1_nodes], axis=1)

            # Calculate the dvalue for the nodes in group2
            dvalue_list[group2_nodes] = np.sum(environment.edge_matrix[group2_nodes][:, group1_nodes], axis=1) - np.sum(
                environment.edge_matrix[group2_nodes][:, group2_nodes], axis=1)

            while environment.group_matrix[group1, group2] == 1:
                print("Group1: ", group1, "Group2: ", group2)

                total_gain = 0
                buffer = []
                labels_list_copy = deepcopy(environment.labels_list)

                # Nodes going from group1 to group2
                x_star = []
                # Nodes going from group2 to group1
                y_star = []

                # find the group of minimum size
                threshold = min(len(group1_nodes), len(group2_nodes))
                # Exhaust every point INTERCHANGE
                for _ in range(threshold):
                    # Find the max gain
                    group1_nodes, group2_nodes = np.meshgrid(
                        group1_nodes, group2_nodes)
                    # Calculate the gain for each combination of nodes
                    gains = dvalue_list[group1_nodes] + dvalue_list[group2_nodes] - \
                        2 * environment.edge_matrix[group1_nodes, group2_nodes]
                    # Get the maximum gain and the indices of the nodes that give the maximum gain
                    node1, node2 = np.unravel_index(
                        np.argmax(gains), gains.shape)
                    gain = gains[node1, node2]
                    node1, node2 = group1_nodes[node1,
                                                node2], group2_nodes[node1, node2]

                    x_star.append(node1)
                    y_star.append(node2)
                    
                    # Remove the nodes from consideration for the next iteration
                    labels_list_copy[node1] = EXCLUDE
                    labels_list_copy[node2] = EXCLUDE

                    # Extract the indices of the nodes in each group
                    group1_nodes = np.where(labels_list_copy == group1)[0]
                    group2_nodes = np.where(labels_list_copy == group2)[0]

                    # Initialize the dvalue_list with zeros
                    dvalue_list = np.zeros(len(environment.xy))

                    # Calculate the dvalue for the nodes in group1
                    dvalue_list[group1_nodes] = np.sum(environment.edge_matrix[group1_nodes][:, group2_nodes], axis=1) - np.sum(
                        environment.edge_matrix[group1_nodes][:, group1_nodes], axis=1)

                    # Calculate the dvalue for the nodes in group2
                    dvalue_list[group2_nodes] = np.sum(environment.edge_matrix[group2_nodes][:, group1_nodes], axis=1) - np.sum(
                        environment.edge_matrix[group2_nodes][:, group2_nodes], axis=1)

                    # update the total gain
                    total_gain += gain
                    buffer.append(
                        {'total_gain': total_gain, 'x_star': x_star.copy(), 'y_star': y_star.copy()})
                    
                max_total_gain = max(buffer, key=lambda x: x['total_gain'])
                if max_total_gain['total_gain'] > 0:
                    INTERCHANGE = True
                    environment.labels_list[max_total_gain['x_star']] = group2
                    print(len(np.where(environment.labels_list == EXCLUDE)[0]))
                    environment.labels_list[max_total_gain['y_star']] = group1
                    environment.group_matrix = _find_group_matrix(environment)
                else:
                    break
        environment.group_matrix = _find_group_matrix(environment)
        return environment, INTERCHANGE


class OneWayPartitioning:
    
    def fit(self, environment: Environment) -> Environment:
        num_clusters = environment.group_matrix.shape[0]
        pairs = [(i, j) for i in range(num_clusters) for j in range(
            num_clusters) if environment.group_matrix[i, j] == 1]
        while pairs:
            i, j = pairs.pop()
            environment, INTERCHANGE = self._one_way_partitioing_single_dvalue(
                i, j, environment)
            if INTERCHANGE:
                # Group j ate all the nodes in group i
                if num_clusters > environment.group_matrix.shape[0]:
                    pairs = [(i, j) for i in range(environment.group_matrix.shape[0]) for j in range(
                        i+1, environment.group_matrix.shape[0]) if environment.group_matrix[i, j] == 1]
                else:
                    for idx in range(len(environment.group_matrix[i])):
                        if idx != i and environment.group_matrix[i, idx] == 1:
                            pairs.append((i, idx))
                            pairs.append((idx, i))
                    for idx in range(len(environment.group_matrix[j])):
                        if idx != j and idx != i and environment.group_matrix[j, idx] == 1:
                            pairs.append((j, idx))
                            pairs.append((idx, j))
        return environment

    # NOT USED ANYMORE
    def _one_way_partitioing_single(self, group1: int, group2: int, environment: Environment):
        """
        Performs a one way partitioning on two groups, group2 takes nodes form group1 given a heuristic.
        Hueristic: select the point with the highest Dvalue and the smallest cluster size.

        :param group1: The index of the first group
        :param group2: The index of the second group
        :param xy: The data points
        :param environment.group_matrix: The matrix that keeps track of the connections between the groups
        :param environment.labels_list: The list that keeps track of the group of each node
        :param environment.edge_matrix: The matrix that keeps track of the connections between the nodes
        :param original_k: The original number of groups: to track the original size because group matrix keeps changing.
        :param margin: The margin of error for the size of the groups
        """

        # If the two clustes are not connected, return
        if environment.group_matrix[group1, group2] == 0:
            return environment.group_matrix, environment.labels_list, False

        else:

            # Performs depth first search to find the set of nodes forming the graph
            def dfs(node, environment):
                """
                Performs depth first search to find the set of nodes forming the graph

                :param node: The node to start from
                :param environment.edge_matrix: The matrix that keeps track of the connections between the nodes
                :param environment.labels_list: The list that keeps track of the group of each node
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
                        connected_nodes = set(
                            np.where(environment.edge_matrix[vertex] == 1)[0])
                        connected_nodes = set(
                            [node for node in connected_nodes if environment.labels_list[node] == environment.labels_list[vertex]])
                        stack.extend(connected_nodes - visited)
                return list(visited)

            # Flag to check if the partitioning was successful
            INTERCHANGE = False
            # The upper bound for the size of the group, original_k because the environment.group_matrix keeps changing and we want to keep track of the original size.
            DEFAULT_SIZE = len(environment.xy) / original_k
            # The limit in which a cluster whould be within
            UPPER_BOUND = (1+self.marin) * DEFAULT_SIZE

            # Extract the indices of the nodes in each group
            group1_nodes = np.where(environment.labels_list == group1)[0]
            group2_nodes = np.where(environment.labels_list == group2)[0]

            # only nodes from group1_nodes that are connected to group2_nodes are considered
            group1_edge_nodes = [i for i in group1_nodes if np.sum(
                environment.edge_matrix[i, environment.labels_list == group2]) > 0]

            # Set the threshold for the loop
            threshold = int(UPPER_BOUND - len(group2_nodes))

            # Compute dvalues
            dvalue_list = np.zeros(len(environment.xy))
            for i in group1_edge_nodes:
                dvalue_list[i] = np.sum(environment.edge_matrix[i, environment.labels_list == group2]) - np.sum(
                    environment.edge_matrix[i, environment.labels_list == group1])

            # Initialize the labels_list_copy that will be used in the loop
            labels_list_copy = environment.labels_list.copy()

            # Loop until the upper bound is reached, or both groups are no longer connected
            while len(group2_nodes) < UPPER_BOUND and environment.group_matrix[group1, group2] == 1:
                # ئreate a list that contains every node conntected node from the same group
                group1_connected_nodes = []

                for i in group1_edge_nodes:
                    group = dfs(i, environment.edge_matrix, labels_list_copy)
                    group1_connected_nodes.append((i, group))

                # Select the node with highest dvalue and smallest conntect group.
                # Reasoning: largest size will guarntee removing a single firewall, bunch of small ones could remove more.
                group1_connected_nodes = [
                    i for i in group1_connected_nodes if len(i[1]) <= threshold]
                selected_group = max(group1_connected_nodes, key=lambda x: (
                    -1*len(x[1]), dvalue_list[x[0]]))[1] if group1_connected_nodes else None

                # If there is a valid group, interchange the labels
                if selected_group:
                    labels_list_copy[selected_group] = group2
                    group1_nodes = np.where(labels_list_copy == group1)[0]
                    group2_nodes = np.where(labels_list_copy == group2)[0]

                    # Update the threshold after the interchange
                    threshold = int(UPPER_BOUND - len(group2_nodes))

                    # only nodes from group1_nodes that are connected to group2_nodes are considered
                    group1_edge_nodes = [i for i in group1_nodes if np.sum(
                        environment.edge_matrix[i, labels_list_copy == group2]) > 0]

                    # Incase group2 ate all the nodes from group1
                    # Find all the group numbers larger than group1 and decrease them by 1, to maintain consecutive numbering
                    if len(group1_nodes) == 0:
                        # Find the group numbers larger than group1
                        group_numbers = np.arange(
                            group1+1, environment.group_matrix.shape[0])
                        # Decrease the group numbers by 1
                        labels_list_copy[np.isin(
                            labels_list_copy, group_numbers)] -= 1

                        # update the group matrix
                        groups_count = len(np.unique(labels_list_copy))
                        environment.group_matrix = np.zeros(
                            (groups_count, groups_count))
                        for i in range(groups_count):
                            for j in range(i + 1, groups_count):
                                if np.sum(environment.edge_matrix[labels_list_copy == i][:, labels_list_copy == j]) > 0:
                                    environment.group_matrix[i, j] = 1
                                    environment.group_matrix[j, i] = 1
                        INTERCHANGE = True
                        break

                    # Update the group matrix
                    groups_count = len(np.unique(labels_list_copy))
                    environment.group_matrix = np.zeros(
                        (groups_count, groups_count))
                    for i in range(groups_count):
                        for j in range(i + 1, groups_count):
                            if np.sum(environment.edge_matrix[labels_list_copy == i][:, labels_list_copy == j]) > 0:
                                environment.group_matrix[i, j] = 1
                                environment.group_matrix[j, i] = 1

                    # Compute dvalues for the next iteration
                    dvalue_list = np.zeros(len(environment.xy))
                    for i in group1_nodes:
                        dvalue_list[i] = np.sum(environment.edge_matrix[i, labels_list_copy == group2]) - np.sum(
                            environment.edge_matrix[i, labels_list_copy == group1])

                    INTERCHANGE = True
                else:
                    break

            return environment.group_matrix, labels_list_copy, INTERCHANGE

    def _one_way_partitioing_single_dvalue(self, group1: int, group2: int, environment: Environment):
        # Flag to check if the partitioning was successful
        INTERCHANGE = False
        UPPER_BOUND = environment.max_size

        # Extract the indices of the nodes in each group
        group1_nodes = np.where(environment.labels_list == group1)[0]
        group2_nodes = np.where(environment.labels_list == group2)[0]

        # Only nodes from group1_nodes that are connected to group2_nodes are considered
        group1_edge_nodes = [node for node in group1_nodes if np.sum(
            environment.edge_matrix[node, environment.labels_list == group2]) > 0]
        # Compute dvalues
        dvalue_list = np.zeros(len(environment.xy))
        for edge_node in group1_edge_nodes:
            dvalue_list[edge_node] = np.sum(environment.edge_matrix[edge_node, environment.labels_list == group2]) - \
                np.sum(
                    environment.edge_matrix[edge_node, environment.labels_list == group1])

        # Initialize the labels_list_copy that will be used in the loop
        labels_list_copy = environment.labels_list.copy()

        total_gain = 0
        buffer = []
        x_star = []
        while len(group2_nodes) < UPPER_BOUND and environment.group_matrix[group1, group2] == 1:
            
            # Find node of highest dvalue from group1
            max_dvalue_node = np.argmax(dvalue_list[group1_edge_nodes])
            max_dvalue_node = group1_edge_nodes[max_dvalue_node]

            # Buffer the node and the gain
            total_gain += dvalue_list[max_dvalue_node]
            x_star.append(max_dvalue_node)
            buffer.append({'gain': total_gain, 'x_star': x_star.copy()})

            # Group2 eats the node
            labels_list_copy[max_dvalue_node] = group2

            # Update the group1_nodes and group2_nodes
            group1_nodes = np.where(labels_list_copy == group1)[0]
            group2_nodes = np.where(labels_list_copy == group2)[0]

            # Only nodes from group1_nodes that are connected to group2_nodes are considered
            group1_edge_nodes = [i for i in group1_nodes if np.sum(
                environment.edge_matrix[i, labels_list_copy == group2]) > 0]

            if len(group1_nodes) == 0:
                # Find the group numbers larger than group1
                group_numbers = np.arange(
                    group1+1, environment.group_matrix.shape[0])
                # Decrease the group numbers by 1
                labels_list_copy[np.isin(labels_list_copy, group_numbers)] -= 1

                # update the group matrix
                groups_count = len(np.unique(labels_list_copy))
                environment.group_matrix = _find_group_matrix(environment=environment)
                INTERCHANGE = True
                break

            # Compute dvalues
            dvalue_list = np.zeros(len(environment.xy))
            for i in group1_edge_nodes:
                dvalue_list[i] = np.sum(environment.edge_matrix[i, labels_list_copy == group2]) - np.sum(
                    environment.edge_matrix[i, labels_list_copy == group1])

            # Update the group matrix
            groups_count = len(np.unique(labels_list_copy))
            environment.group_matrix = _find_group_matrix(environment=environment)

        if buffer:
            # Find the best buffer
            best_buffer = max(buffer, key=lambda x: x['gain'])
            if best_buffer['gain'] > 0:
                # Interchange the nodes
                for node in best_buffer['x_star']:
                    environment.labels_list[node] = group2
                INTERCHANGE = True
            # Update the group matrix

        # TODO: Every time we interchange, we need to update the group matrix. this is done above, but for some reason I should update it here as well
        environment.group_matrix = _find_group_matrix(environment=environment)
        return environment, INTERCHANGE


class Bisection:
    def __init__(self, max_firewalls: int, variation: str = 'self'):
        self.variation = variation
        self.max_firewalls = max_firewalls

    def fit(self, environment: Environment, return_runs: bool = False):
        multi_way = MultiwayPartioning()
        one_way = OneWayPartitioning()

        max_size_pool = np.arange(5, environment.max_size, 5)
        buffer = []

        # Start bisecting
        high = len(max_size_pool)-1
        low = 0

        while high - low > 1:
            mid = (high + low) // 2
            environment_copy = deepcopy(environment)
            environment_copy.max_size = max_size_pool[mid]
            environment_copy = multi_way.fit(environment_copy)
            environment_copy = one_way.fit(environment_copy)

            if self.variation == 'self':
                firewall_count = len(
                    environment_copy.find_firewalls(variation="self"))
            elif self.variation == 'other':
                firewall_count = len(
                    environment_copy.find_firewalls(variation="other")[0])

            if firewall_count > self.max_firewalls:
                low = mid
            else:
                high = mid

            buffer.append({
                'max_size': max_size_pool[mid],
                'firewalls': firewall_count,
                'environment': environment_copy
            })

        # Return the best run: best run is the the minimum(max_size) given the maximum number of firewalls that is less than or equal the max_firewalls
        best_run = min([i for i in buffer if i['firewalls'] <=
                       self.max_firewalls], key=lambda x: x['max_size'])

        if return_runs:
            return best_run['environment'], buffer
        else:
            return best_run['environment']


if __name__ == "__main__":

    num_nodes = 500
    connect_threshold = 1
    num_clusters = 5
    xMin = 0
    xMax = 20
    yMin = 0
    yMax = 20
    x = np.random.uniform(size=num_nodes, low=xMin, high=xMax)
    y = np.random.uniform(size=num_nodes, low=yMin, high=yMax)
    xy = np.array([x, y]).T.astype('float32')
