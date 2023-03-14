
import json
import multiprocessing as mp
import time
from GraPart.partition import MultiwayPartioning, OneWayPartitioning, Bisection
from GraPart.environment import Environment
import numpy as np
import os

SAVE_PATH = 'Benchmark_results/'


def one_way_PLUS_multi_way_bench(environment: Environment) -> dict:
    """
    This function benchmarks the one way partitioning
    """
    # This function benchmarks the one way partitioning
    results = {}
    firewalls_self_before = len(environment.find_firewalls(variation="self"))
    firewalls_other_before = len(
        environment.find_firewalls(variation="other")[0])

    start_multiway = time.time()
    environment = MultiwayPartioning().fit(environment)
    end_multiway = time.time()

    firewalls_self_multiway = len(environment.find_firewalls(variation="self"))
    firewalls_other_multiway = len(
        environment.find_firewalls(variation="other")[0])

    start_oneway = time.time()
    environment = OneWayPartitioning().fit(environment)
    end_oneway = time.time()

    firewalls_self_oneway = len(environment.find_firewalls(variation="self"))
    firewalls_other_oneway = len(
        environment.find_firewalls(variation="other")[0])

    results['number_of_nodes'] = len(environment.xy)
    results['max_size'] = environment.max_size
    results['number_of_edges'] = len(np.where(environment.edge_matrix == 1)[0])
    results['KMeans'] = {
        "firewalls_self": firewalls_self_before,
        "firewalls_other": firewalls_other_before,
    }
    results["multiway"] = {
        "time": (end_multiway - start_multiway) * 1000,
        "firewalls_self": firewalls_self_multiway,
        "firewalls_other": firewalls_other_multiway,
    }
    results["oneway"] = {
        "time": (end_oneway - start_oneway) * 1000,
        "firewalls_self": firewalls_self_oneway,
        "firewalls_other": firewalls_other_oneway,
    }

    return results

def one_way_PLUS_multi_way_bench(environment: Environment) -> dict:
    """
    This function benchmarks the one way partitioning
    """

    results = {}
    firewalls_self_before = len(environment.find_firewalls(variation="self"))
    firewalls_other_before = len(
        environment.find_firewalls(variation="other")[0])

    start_multiway = time.time()
    environment = MultiwayPartioning().fit(environment)
    end_multiway = time.time()

    firewalls_self_multiway = len(environment.find_firewalls(variation="self"))
    firewalls_other_multiway = len(
        environment.find_firewalls(variation="other")[0])

    start_oneway = time.time()
    environment = OneWayPartitioning().fit(environment)
    end_oneway = time.time()

    firewalls_self_oneway = len(environment.find_firewalls(variation="self"))
    firewalls_other_oneway = len(
        environment.find_firewalls(variation="other")[0])

    results['number_of_nodes'] = len(environment.xy)
    results['max_size'] = environment.max_size
    results['number_of_edges'] = len(np.where(environment.edge_matrix == 1)[0])
    results['KMeans'] = {
        "firewalls_self": firewalls_self_before,
        "firewalls_other": firewalls_other_before,
    }
    results["multiway"] = {
        "time": (end_multiway - start_multiway) * 1000,
        "firewalls_self": firewalls_self_multiway,
        "firewalls_other": firewalls_other_multiway,
    }
    results["oneway"] = {
        "time": (end_oneway - start_oneway) * 1000,
        "firewalls_self": firewalls_self_oneway,
        "firewalls_other": firewalls_other_oneway,
    }

    return results


def parallel_benchmark_multiway_oneway(
        n_nodes: int,
        connect_threshold: float,
        x_max: float,
        y_max: float,
        max_iter: int = 1000,
        save_dir: str = None) -> None:

    if not os.path.exists(f'{SAVE_PATH}{save_dir}/raw_results'):
        os.makedirs(f'{SAVE_PATH}{save_dir}/raw_results')

    # Divide the number of clusters into multiple processes
    iterations_per_process = max_iter // mp.cpu_count()
    processes = []

    # Create a process for each set of clusters
    for process_id in range(mp.cpu_count()):
        print('A new process (ID = {}) will run {} iterations'.format(
            process_id, iterations_per_process))
        # What is wrong the following line?
        # answer:
        p = mp.Process(target=process_benchmark_multiway_oneway, args=(process_id,
                                                                       n_nodes,
                                                                       connect_threshold,
                                                                       x_max,
                                                                       y_max,
                                                                       iterations_per_process,
                                                                       save_dir
                                                                       ))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()


def process_benchmark_multiway_oneway(process_id: str,
                                      n_nodes: int,
                                      connect_threshold: float,
                                      x_max: float,
                                      y_max: float,
                                      max_iter: int,
                                      save_dir: str):
    results = []
    for _ in range(max_iter):
        x = np.random.uniform(size=n_nodes, low=0, high=x_max)
        y = np.random.uniform(size=n_nodes, low=0, high=y_max)
        xy = np.array([x, y]).T.astype('float32')
        max_size_pool = np.arange(10, len(xy)/2, 10)
        for max_size in max_size_pool:
            # try:
                print("Running benchmark for max_size = {}".format(max_size))
                environment = Environment(xy, max_size, connect_threshold)
                res = one_way_PLUS_multi_way_bench(environment)
                results.append(res)
            # except Exception as e:
            #     print("Error in benchmarking for {} max_size".format(max_size))
            #     print(e)
            #     continue

    with open(f'{SAVE_PATH}{save_dir}/raw_results/results_{process_id}.json', 'w') as f:
        json.dump(results, f)


# -------------------------------BENCHMARKS FOR BISECTION--------------------------------------------
def bisection_bench(environment: Environment, max_firewalls: int, variation: str) -> dict:
    """
    This function benchmarks the one way partitioning
    """

    # Save the time of the start of the function
    start = time.time()

    # Call the bisection function
    environment, runs = Bisection(max_firewalls=max_firewalls).fit(environment)
    # Save the time of the end of the function
    end = time.time()

    # Select the best partitioning for the given number of firewalls
    eligible = [i for i in runs if i['firewalls'] <= max_firewalls]
    selected = max(eligible, key=lambda x: x['clusters'])[
        'clusters'] if eligible else 'max_firewalls not enough'

    # Save the results
    results = {}
    results['number_of_nodes'] = len(environment.xy)
    results['max_firewalls'] = max_firewalls
    results['results'] = runs
    results['best_run'] = selected
    results['time'] = (end - start) * 1000

    # Return the results
    return results


def parallel_benchmark_bisection(max_firewalls: int,
                                 n_nodes: int,
                                 connect_threshold: float,
                                 x_max: float,
                                 y_max: float,
                                 variation: str,
                                 max_iter: int = 1000,
                                 save_dir: str = None):

    if not os.path.exists(f'{SAVE_PATH}{save_dir}/raw_results'):
        os.makedirs(f'{SAVE_PATH}{save_dir}/raw_results')

    # Divide the number of clusters into multiple processes
    iterations_per_process = max_iter // mp.cpu_count()
    processes = []

    # Create a process for each set of clusters
    for process_id in range(mp.cpu_count()):
        print('A new process (ID = {}) will run {} iterations'.format(
            process_id, iterations_per_process))
        p = mp.Process(target=process_benchmark_bisection, args=(process_id,
                                                                 max_firewalls,
                                                                 n_nodes,
                                                                 connect_threshold,
                                                                 x_max,
                                                                 y_max,
                                                                 variation,
                                                                 iterations_per_process,
                                                                 save_dir))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()


def process_benchmark_bisection(process_id: str,
                                max_firewalls: int,
                                n_nodes: int,
                                connect_threshold: float,
                                x_max: float,
                                y_max: float,
                                variation: str,
                                max_iter: int,
                                save_dir: str):
    results = []
    for _ in range(max_iter):
        print("Running benchmark for {} firewalls".format(num_firewalls))
        x = np.random.uniform(size=n_nodes, low=0, high=x_max)
        y = np.random.uniform(size=n_nodes, low=0, high=y_max)
        xy = np.array([x, y]).T.astype('float32')
        environment = Environment(xy, 0, connect_threshold)
        for num_firewalls in range(0, max_firewalls):
            try:
                res = bisection_bench(environment=environment,
                                      max_firewalls=num_firewalls,
                                      variation=variation)
                results.append(res)
                # how can i print the error that happened here in the except?
                # answer:
            except Exception as e:
                print("Failed to run benchmark for {} firewalls".format(num_firewalls))
                print(e)
                continue

    with open(f'{SAVE_PATH}{save_dir}/raw_results/results_{process_id}.json', 'w') as f:
        json.dump(results, f)
