import numpy as np
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass
from k_means_constrained import KMeansConstrained

@dataclass
class Environment:
    xy: np.ndarray
    _max_size: int
    connect_threshold: float
    isolated_clusters: bool = True
    group_matrix: np.ndarray = None
    labels_list: np.ndarray = None
    edge_matrix: np.ndarray = None

        
    @property
    def max_size(self) -> int:
        return self._max_size
    
    @max_size.setter
    def max_size(self, value: int) -> None:
        self._max_size = value
        self.__post_init__(isolated_clusters=True)

    def __post_init__(self, isolated_clusters=True) -> None:
        """
        Setup the environment for the graph partitioning problem.
        """
        xy_copy = self.xy.copy()

        # Calculate the pairwise distances between all points
        distances = np.linalg.norm(self.xy[:, None, :] - self.xy, axis=-1)
        self.edge_matrix = distances <= self.connect_threshold
        np.fill_diagonal(self.edge_matrix, 0)
        self.edge_matrix = self.edge_matrix.astype(int)

        # If isolated_clusters is True, then create isolated clusters
        if isolated_clusters:
            # Nodes that are not in the isloated clusters will be labeled with -1
            labels_list_isolated = self._create_isolated_clusters()
            xy_copy = xy_copy[labels_list_isolated == -1]
            
            # If everthing is isolated, then return
            if len(xy_copy) == 0:
                self.labels_list = labels_list_isolated
                
                clusters = np.unique(self.labels_list)
                group_matrix = np.zeros((len(clusters), len(clusters)))
                for cluster1 in range(len(clusters)):
                    cluster1_nodes_idx = np.where(self.labels_list == clusters[cluster1])[0]
                    for cluster2 in range(cluster1+1, len(clusters)):
                        cluster2_nodes_idx = np.where(self.labels_list == clusters[cluster2])[0]
                        if np.sum(self.edge_matrix[cluster1_nodes_idx][:, cluster2_nodes_idx]) > 0:
                            group_matrix[cluster1][cluster2] = 1
                            group_matrix[cluster2][cluster1] = 1
                    
                self.group_matrix = group_matrix
                return 
        
        remaining_nodes = len(xy_copy)
        n_clusters = int(np.ceil(remaining_nodes/self.max_size))

        # Select the centroids for the KMeans algorithm
        centroids = self._select_KMeans_centroids(n_clusters)
        # Perform the KMeans clustering
        self.labels_list = KMeansConstrained(
            n_clusters=n_clusters, size_max=self.max_size, init=centroids).fit(xy_copy).labels_

        if isolated_clusters:
            idx = 0
            compensation = len(np.unique(labels_list_isolated)) - 1
            for i, label in enumerate(labels_list_isolated):
                if label == -1:
                    labels_list_isolated[i] = self.labels_list[idx] + compensation
                    idx += 1
            self.labels_list = labels_list_isolated

        # Create the group matrix
        clusters = np.unique(self.labels_list)
        group_matrix = np.zeros((len(clusters), len(clusters)))
        for cluster1 in range(len(clusters)):
            cluster1_nodes_idx = np.where(self.labels_list == clusters[cluster1])[0]
            for cluster2 in range(cluster1+1, len(clusters)):
                cluster2_nodes_idx = np.where(self.labels_list == clusters[cluster2])[0]
                if np.sum(self.edge_matrix[cluster1_nodes_idx][:, cluster2_nodes_idx]) > 0:
                    group_matrix[cluster1][cluster2] = 1
                    group_matrix[cluster2][cluster1] = 1
                    
        self.group_matrix = group_matrix
                
    def _select_KMeans_centroids(self, n_clusters: int) -> np.ndarray:
        """
        This function is used to select the centroids for the KMeans algorithm
        n_clusters: the number of clusters

        :return: the centroids: (n_clusters,2) matrix
        """

        # Select the first centroid randomly
        centroids = np.random.choice(len(self.xy), 1)
        # Select the next centroid as the point that is the furthest from the current centroid
        for i in range(1, n_clusters):
            distances = np.linalg.norm(
                self.xy[centroids[i-1]] - self.xy, axis=-1)
            centroids = np.append(centroids, np.argmax(distances))
        return self.xy[centroids.astype(int)]

    def _create_isolated_clusters(self) -> np.ndarray:
        """
        This function is used to create isolated clusters using DFS.
        Isolated clusters are a set of nodes that form a cluster below a certain size.
        Isoalted are not considered in the paritioning to make the computation faster.

        :return: the labels of the nodes: (num_nodes,1) matrix
        """
        num_nodes = len(self.xy)
        clusters = []
        visited = [False] * num_nodes
        for node in range(num_nodes):
            if not visited[node]:
                cluster = []
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if not visited[curr]:
                        visited[curr] = True
                        cluster.append(curr)
                        for neighbor in range(num_nodes):
                            if self.edge_matrix[curr][neighbor]:
                                stack.append(neighbor)
                clusters.append(cluster)

            clusters = [cluster for cluster in clusters if len(
                cluster) <= self.max_size]

        # intitilzie labels_list iwht -1 using numpy
        labels_list = np.full(num_nodes, -1, dtype=int)
        for label , cluster in enumerate(clusters):
            for node in cluster:
                labels_list[node] = label

        return labels_list

    # Utils: Plotting
    def plot(self) -> plt.Axes:
        """
        This function is used to plot the graph

        return: the figure of the graph
        """
        fig , ax = plt.subplots()
        ax = self._plot_nodes(ax)
        ax = self._plot_edges(ax)
        ax = self._plot_firewalls(ax)
        return fig,ax

    def _plot_nodes(self, ax: plt.Axes) -> plt.Axes:

        """
        This function is used to plot the nodes
        ax: the figure

        return: the figure of the nodes
        """

        # plot the nodes
        ax.scatter(self.xy[:, 0], self.xy[:, 1], c=self.labels_list)
        return ax

    def _plot_edges(self, ax: plt.Axes) -> plt.Axes:
        """
        This function is used to plot the edges
        ax: the figure

        return: the figure of the edges
        """
        # plot the edges
        for i in range(len(self.xy)):
            for j in range(len(self.xy)):
                if self.edge_matrix[i, j] > 0:
                    ax.plot([self.xy[i, 0], self.xy[j, 0]], [
                            self.xy[i, 1], self.xy[j, 1]], 'k-')
        return ax

    def _plot_firewalls(self, ax: plt.Axes, variation: str = "self", color: str = "green", color2: str = 'orange') -> plt.Axes:
        """
        This function is used to plot the firewalls
        ax: the figure
        variation: the variation of the firewall (self, other)
        color: the color of the firewall
        color2: the color of the protected nodes - if variation is other

        return: the figure of the firewalls
        """

        if variation == "self":
            firewall_nodes = self.find_firewalls(variation)
            ax.scatter(firewall_nodes[:, 0], firewall_nodes[:, 1], c=color)
            return ax
        elif variation == "other":
            firewall_nodes, protected_nodes = self. find_firewalls(variation)
            ax.scatter(firewall_nodes[:, 0], firewall_nodes[:, 1], c=color)
            ax.scatter(protected_nodes[:, 0], protected_nodes[:, 1], c=color2)
            return ax

    # Utils: Firewall
    def find_firewalls(self, variation: str ="self") -> np.ndarray:
        """
        This function is used to find the firewalls
        variation: the variation of the firewall (self, other)

        """

        states = np.zeros(len(self.xy))
        if variation == "self":
            return self._find_firewall_self(states)
        elif variation == "other":
            return self._find_firewall_other(states)
        else:
            raise ValueError("Variation must be either 'self' or 'other'.")

    def _find_firewall_self(self, states: np.ndarray) -> np.ndarray:
        """
        This function is used to find the firewalls using the self variation
        states: the states of the nodes (0-normal node, 1-firewall, 2-protected node)

        return: the coordinates of the firewalls
        """
        num_clusters = len(np.unique(self.labels_list))
        num_nodes = len(self.labels_list)
        for i in range(num_nodes):
            connections = np.zeros(num_clusters)
            for j in range(num_nodes):
                if self.edge_matrix[i, j] == 1 and states[j] != 1:
                    connections[self.labels_list[j]] += 1
            connections[self.labels_list[i]] = 0
            if np.sum(connections) > 0:
                states[i] = 1
        return self.xy[states == 1]

    def _find_firewall_other(self, states: np.ndarray):
        """
        This function is used to find the firewalls using the other variation
        states: the states of the nodes (0-normal node, 1-firewall, 2-protected node)

        return: the coordinates of the firewalls and the protected nodes
        """
        num_clusters = len(np.unique(self.labels_list))
        num_nodes = len(self.labels_list)
        for i in range(num_nodes):
            if states[i] == 1 or states[i] == 2:
                continue
            connections = np.zeros(num_clusters)
            for j in range(num_nodes):
                if self.edge_matrix[i, j] == 1 and states[j] != 1 and states[j] != 2:
                    connections[self.labels_list[j]] += 1
            connections[self.labels_list[i]] = 0
            if np.sum(connections) > 0:
                states[self.edge_matrix[i, :] == 1] = 2
                states[i] = 1

        return self.xy[states == 1], self.xy[states == 2]

    def deep_copy(self):
        """
        This function is used to create a deep copy of the graph
        """
        return copy.deepcopy(self)
    
    def __repr__(self) -> str:
        return "a Graph object"
    
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Environment):
            return False
        return np.array_equal(self.edge_matrix, __o.edge_matrix) and np.array_equal(self.xy, __o.xy) and np.array_equal(self.labels_list, __o.labels_list)