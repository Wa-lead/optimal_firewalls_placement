import numpy as np
from k_means_constrained import KMeansConstrained

def select_KMeans_centroids(xy, num_clusters):
    # select the centroids for the KMeans algorithm
    # select the first centroid randomly
    centroids = np.random.choice(len(xy), 1)
    # select the next centroid as the point that is the furthest from the current centroid
    for i in range(1, num_clusters):
        distances = np.linalg.norm(xy[centroids[i-1]] - xy, axis=-1)
        centroids = np.append(centroids, np.argmax(distances))
    return xy[centroids.astype(int)]

def setup(xy, num_clusters , connect_distance):
    # create connect_matrix, labels_list, and num_groups
    # Calculate the pairwise distances between all points
    distances = np.linalg.norm(xy[:, None, :] - xy, axis=-1)
    # Set the elements of the connect_matrix to 1 if the distance is less than or equal to the connect_distance
    connect_matrix = distances <= connect_distance
    # zero the diagonal
    np.fill_diagonal(connect_matrix, 0)
    connect_matrix = connect_matrix.astype(int)


    up_bound = int(1.1*(len(xy)/num_clusters))
    low_bound = int(0.9*(len(xy)/num_clusters))
    centroids = select_KMeans_centroids(xy, num_clusters)
    labels_list= KMeansConstrained(n_clusters=num_clusters, size_min=low_bound, size_max=up_bound, init=centroids).fit(xy).labels_

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
