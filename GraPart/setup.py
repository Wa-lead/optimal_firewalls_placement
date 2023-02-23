import numpy as np
from k_means_constrained import KMeansConstrained
import matplotlib.pyplot as plt

def select_KMeans_centroids(xy, num_clusters):
    # select the centroids for the KMeans algorithm
    # select the first centroid randomly
    centroids = np.random.choice(len(xy), 1)
    # select the next centroid as the point that is the furthest from the current centroid
    for i in range(1, num_clusters):
        distances = np.linalg.norm(xy[centroids[i-1]] - xy, axis=-1)
        centroids = np.append(centroids, np.argmax(distances))
    return xy[centroids.astype(int)]


def create_isolated_clusters(connect_matrix, xy, num_clusters, low_bound, up_bound):
    num_nodes = len(xy)
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
                        if connect_matrix[curr][neighbor]:
                            stack.append(neighbor)
            clusters.append(cluster)

        clusters = [cluster for cluster in clusters if len(cluster) <= up_bound and len(cluster) >= low_bound]

    # intitilzie labels_list iwht -1 using numpy
    labels_list = np.full(num_nodes, -1, dtype=int)
    for i, cluster in enumerate(clusters):
        for node in cluster:
            labels_list[node] = i

    return labels_list

def setup(xy, num_clusters , connect_distance, isolated_clusters=True):

    # Calculate the pairwise distances between all points
    distances = np.linalg.norm(xy[:, None, :] - xy, axis=-1)
    # Set the elements of the connect_matrix to 1 if the distance is less than or equal to the connect_distance
    connect_matrix = distances <= connect_distance

    # zero the diagonal
    np.fill_diagonal(connect_matrix, 0)
    connect_matrix = connect_matrix.astype(int)

    # Set the upper and lower bounds for the number of nodes in each cluster
    up_bound = np.ceil(1.1*(len(xy)/num_clusters))
    low_bound = int(0.9*(len(xy)/num_clusters))


    # if isolated_clusters is True, then create isolated clusters
    if isolated_clusters:
        # Nodes that are not in the isloated clusters will be labeled with -1
        labels_list_isolated = create_isolated_clusters(connect_matrix, xy, num_clusters, low_bound, up_bound)
        xy = xy[labels_list_isolated == -1]
        num_clusters -= len(np.unique(labels_list_isolated)) -1

    # Reset the boundaries after the first clustering step
    up_bound = np.ceil(1.1*(len(xy)/num_clusters))
    low_bound = int(0.9*(len(xy)/num_clusters))

    # log for debugging
    print("Number of clusters: ", num_clusters)
    print("Upper bound: ", up_bound)
    print('size_max * num_clusters: ', up_bound * num_clusters)
    print('number of nodes: ', len(xy)) 
    centroids = select_KMeans_centroids(xy, num_clusters)
    labels_list= KMeansConstrained(n_clusters=num_clusters, size_min=low_bound, size_max=up_bound, init=centroids).fit(xy).labels_
    
    if isolated_clusters:
        # Compensate for the number of isolated clusters
        cluster_count_compensation = np.unique(labels_list_isolated).shape[0] - 1
        idx = 0
        for i, label in enumerate(labels_list_isolated):
            if label == -1:
                labels_list_isolated[i] = labels_list[idx] + cluster_count_compensation
                idx += 1
        labels_list = labels_list_isolated


    # Get the unique group labels
    group_labels = np.unique(labels_list)
    # Create a boolean matrix with True at the positions where i != j
    group_matrix = group_labels[:, None] != group_labels
    # Convert the boolean matrix to an integer matrix
    group_matrix = group_matrix.astype(int)


    return connect_matrix, labels_list, group_matrix

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
    xy = np.array([x,y]).T.astype('float32')

    print(select_KMeans_centroids(xy, num_clusters))
