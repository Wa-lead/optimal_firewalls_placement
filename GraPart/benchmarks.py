
import json
import multiprocessing as mp
import time
from GraPart.firewall import find_firewalls
from GraPart.setup import setup
from GraPart.parition_benchmark import multiway_partitioning, one_way_partitioing, bisection
import numpy as np
import os

SAVE_PATH = 'Benchmark_results/'

def one_way_PLUS_multi_way_bench(xy, group_matrix, labels_list, connect_matrix, margin=0.1):
    """
    This function benchmarks the one way partitioning
    """
    num_clusters = len(np.unique(labels_list))

    results = {}
    firewalls_self_before = len(find_firewalls(
        labels_list, connect_matrix, xy, "self"))
    firewalls_other_before = len(find_firewalls(
        labels_list, connect_matrix, xy, "other")[0])

    start = time.time()

    group_matrix, labels_list, count1 = multiway_partitioning(
        xy, group_matrix, labels_list, connect_matrix)
    mid = time.time()

    firewalls_self_multiway = len(find_firewalls(
        labels_list, connect_matrix, xy, "self"))
    firewalls_other_multiway = len(find_firewalls(
        labels_list, connect_matrix, xy, "other")[0])

    group_matrix, labels_list, count2 = one_way_partitioing(
        xy, group_matrix, labels_list, connect_matrix, margin= margin)
    end = time.time()

    firewalls_self_oneway = len(find_firewalls(
        labels_list, connect_matrix, xy, "self"))
    firewalls_other_oneway = len(find_firewalls(
        labels_list, connect_matrix, xy, "other")[0])

    results['number_of_nodes'] = len(xy)
    results['number_of_clusters'] = num_clusters
    results['number_of_edges'] = len(np.where(connect_matrix == 1)[0])
    results['KMeans'] = {
        "firewalls_self": firewalls_self_before,
        "firewalls_other": firewalls_other_before,
    }
    results["multiway"] = {
        "time": (mid - start) * 1000,
        "firewalls_self": firewalls_self_multiway,
        "firewalls_other": firewalls_other_multiway,
        "count": count1
    }
    results["oneway"] = {
        "time": (end - mid) * 1000,
        "firewalls_self": firewalls_self_oneway,
        "firewalls_other": firewalls_other_oneway,
        "converge": count2
    }

    return results


def parallel_benchmark_multiway_oneway(max_clusters, num_nodes, connect_threshold, xMax, yMax, max_iter=1000, margin=0.1, save_dir=None):
    if not os.path.exists(f'{SAVE_PATH}{save_dir}/raw_results'):
        os.makedirs(f'{SAVE_PATH}{save_dir}/raw_results')


    # Divide the number of clusters into multiple processes
    firewalls_per_cluster = max_clusters // mp.cpu_count()
    processes = []

    print("Starting benchmark for {} clusters".format(max_clusters))
    print(max_iter, mp.cpu_count())
    # Create a process for each set of clusters
    for i in range(mp.cpu_count()):
        print("Starting process {}".format(i))
        if i == 0:
            start = 2
            end = firewalls_per_cluster
        else:
            start = i * firewalls_per_cluster
            end = start + firewalls_per_cluster
        p = mp.Process(target=process_benchmark_multiway_oneway, args=(
            start, end, num_nodes, connect_threshold, xMax, yMax, max_iter, margin, save_dir))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()


def process_benchmark_multiway_oneway(start, end, num_nodes, connect_threshold, xMax, yMax, max_iter, margin, save_dir):
    print(end - start, max_iter)
    results = []
    for num_clusters in range(start, end):
        for i in range(max_iter):
            print("Running benchmark for {} clusters".format(num_clusters))
            x = np.random.uniform(size=num_nodes, low=0, high=xMax)
            y = np.random.uniform(size=num_nodes, low=0, high=yMax)
            xy = np.array([x, y]).T.astype('float32')
            connect_matrix, labels_list, group_matrix = setup(
                xy, num_clusters, connect_threshold)
            res = one_way_PLUS_multi_way_bench(
                xy, group_matrix, labels_list, connect_matrix, margin=margin)
            results.append(res)

    with open(f'{SAVE_PATH}{save_dir}/raw_results/results_{start}_{time.time()}.json', 'w') as f:
        json.dump(results, f)


# -------------------------------BENCHMARKS FOR BISECTION--------------------------------------------
def bisection_bench(xy,
                    max_clusters=30,
                    connect_distance=1,
                    max_firewalls=50,
                    variation='self',
                    margin=0.1):
    """
    This function benchmarks the one way partitioning
    """

    # Save the time of the start of the function
    start = time.time()

    # Call the bisection function
    result = bisection(xy,
                       max_clusters=max_clusters,
                       connect_distance=connect_distance,
                       max_firewalls=max_firewalls,
                       variation=variation,
                       margin=margin)

    # Save the time of the end of the function
    end = time.time()

    # Select the best partitioning for the given number of firewalls
    eligible = [i for i in result if i['firewalls'] <= max_firewalls]
    selected = max(eligible, key=lambda x: x['clusters'])['clusters'] if eligible else 'max_firewalls not enough'

    # Save the results
    results = {}
    results['number_of_nodes'] = len(xy)
    results['max_firewalls'] = max_firewalls
    results['results'] = result
    results['clusters'] = selected
    results['time'] = (end - start) * 1000

    # Return the results
    return results


def parallel_benchmark_bisection(max_clusters,max_firewalls, num_nodes, connect_threshold, xMax, yMax, variation, margin=0.1,max_iter=1000,save_dir=None):
    if not os.path.exists(f'{SAVE_PATH}{save_dir}/raw_results'):
        os.makedirs(f'{SAVE_PATH}{save_dir}/raw_results')

    # Divide the number of clusters into multiple processes
    firewalls_per_cluster = max_firewalls // mp.cpu_count()
    processes = []

    # Create a process for each set of clusters
    for i in range(mp.cpu_count()):
        start = i * firewalls_per_cluster
        end = start + firewalls_per_cluster
        p = mp.Process(target=process_benchmark_bisection, args=(start, end, max_clusters, num_nodes, connect_threshold, xMax, yMax, variation, max_iter, margin, save_dir))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()


def process_benchmark_bisection(start, end, max_xlusters, num_nodes, connect_threshold, xMax, yMax, variation, max_iter, margin,save_dir):
    results = []
    for num_firewalls in range(start, end):
        for _ in range(max_iter):
            print("Running benchmark for {} firewalls".format(num_firewalls))
            x = np.random.uniform(size=num_nodes, low=0, high=xMax)
            y = np.random.uniform(size=num_nodes, low=0, high=yMax)
            xy = np.array([x, y]).T.astype('float32')
            res = bisection_bench(xy,
                                  max_clusters=max_xlusters,
                                  connect_distance=connect_threshold,
                                  max_firewalls=num_firewalls,
                                  variation=variation,
                                  margin=margin)
            results.append(res)

    with open(f'{SAVE_PATH}{save_dir}/raw_results/results_{start}.json', 'w') as f:
        json.dump(results, f)
