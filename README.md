# About

Adopted from: http://eda.ee.ucla.edu/EE201A-04Spring/kl.pdf

The paper presents a heuristic algorithm for partitioning a graph with 2n vertices into two subsets of n vertices each, with the goal of minimizing the "external cost" T, which is the sum of the costs of connections between the two subsets. The algorithm starts with an arbitrary partition **(KMeans in this implementation**) of the graph and repeatedly interchanges subsets of vertices between the two subsets in an attempt to reduce T. The subsets to be interchanged are chosen using a rule that involves the difference between external and internal costs for each vertex. The algorithm terminates when no further improvement is possible and the resulting partition is considered to be locally minimum. The process can then be repeated with different starting partitions to find multiple locally minimum partitions.

Our proposed variation"one-way partitioning" allows for one-sided gain in the partitioning process. Specifically, this variation allows one subset to selectively absorb nodes from the other subset based on a heuristic that prioritizes the nodes with the highest cost and the smallest graph size within that subset. By allowing for the absorption of small pieces of one subset into the other, this variation aims to minimize the number of firewalls required for the partitioning. However, to prevent the imbalance of the sizes of the subsets, the size of the subset that is absorbing the nodes is restricted by a pre-specified margin. This ensures that the partitioning process results in a balanced distribution of nodes across the subsets.


# Results
![ONEWAY](https://user-images.githubusercontent.com/81301826/212500989-a31129f0-0ed6-4d8b-90be-658274dd208f.png)

# Benchmarks
> NOTE: 'self' and 'other' variation are different vairation of the firewalls. 'self' protects itself only, while 'other' protects itself and its connected nodes. 'other' variation results in less number of required firewalls

### Number of firewalls vs Number of clusters (self)
![self_variation_MO](https://user-images.githubusercontent.com/81301826/212501075-77bc502e-0a28-424b-8268-4809286d4723.png)

### Number of firewalls vs Number of clusters (other)
![other_variation_MO](https://user-images.githubusercontent.com/81301826/212501077-d62129b5-07cc-46ca-ad67-eb2c77226ad9.png)

### Convergence
![converge_MO](https://user-images.githubusercontent.com/81301826/212501020-b338bff0-03c5-45b1-982d-8802aa8951a1.png)

### Execution time (ms)
![time(ms)_MO](https://user-images.githubusercontent.com/81301826/212501095-d7eddd7f-6b9b-4532-bcea-fff8ad21c6e1.png)


### How manu clusters you can achieve given an avaliable number of firewalls
![clusters_given_firewalls](https://user-images.githubusercontent.com/81301826/212501108-4b42ac69-ecd8-4168-8657-77139c21a9ec.png)
