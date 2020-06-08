import mlrose
from typing import List, Set, Tuple
import numpy as np
from MatrixGenerators import ReducedMatrix
from RegretEstimators import DataLossEstimator
import functools


def greedy_search(matrix: np.ndarray, cluster_size: int) -> List[Set[int]]:
    open_nodes = list(range(len(matrix)))
    clusters = []
    cluster = []
    while open_nodes:
        if len(cluster) == cluster_size:
            clusters.append(set(cluster))
            cluster = []
        if not cluster:
            chosen_index = np.random.randint(len(open_nodes))
        else:
            chosen_index = np.argmax([ReducedMatrix.coarse_element(matrix, cluster, n) for n in open_nodes])
        cluster.append(open_nodes.pop(chosen_index))
    if cluster:
        clusters.append(set(cluster))
    return clusters


def greedy_loop(matrix: np.ndarray, cluster_size: int, solver, iter_num: int = 1) -> List[Set[int]]:
    min_regret = np.inf
    best_solution = []
    for i in range(iter_num):
        solution = solver(matrix, cluster_size)
        regret = DataLossEstimator.calculate_data_loss(matrix, solution)
        if regret < min_regret:
            min_regret = regret
            best_solution = solution
    return best_solution


def fix_assignment(assignment_vector: np.ndarray, cluster_size: int) -> List[Set[int]]:
    clusters_dict = dict()
    last_assignment = np.max(assignment_vector) + 1
    exists_and_is_full = lambda c: c in clusters_dict and len(clusters_dict[c]) >= cluster_size
    for node, assignment in enumerate(assignment_vector):
        if exists_and_is_full(assignment):
            if exists_and_is_full(last_assignment):
                last_assignment += 1
            assignment = last_assignment
        if assignment not in clusters_dict:
            clusters_dict[assignment] = set()
        clusters_dict[assignment].add(node)
    return [c for c in clusters_dict.values()]


def fitness_function(assignment_vector: np.ndarray, matrix: np.ndarray, cluster_size: int) -> float:
    valid_clusters = fix_assignment(assignment_vector, cluster_size)
    return np.sum(matrix) - DataLossEstimator.calculate_data_loss(matrix, valid_clusters)


def clusters2assignment_vector(clusters: List[Set[int]]) -> np.ndarray:
    vec = np.zeros(np.sum([len(c) for c in clusters]))
    for i, cluster in enumerate(clusters):
        for member in cluster:
            vec[member] = i
    return vec


def solve(matrix: np.ndarray, cluster_size: int, solver) -> List[Set[int]]:
    kwargs = {'matrix': matrix, 'cluster_size': cluster_size}
    fitness = mlrose.CustomFitness(fitness_function, **kwargs)
    matrix_dimension = matrix.shape[0]
    problem = mlrose.DiscreteOpt(length=matrix_dimension, fitness_fn=fitness, maximize=True, max_val=matrix_dimension)
    best_state, best_fitness = solver(problem)
    return fix_assignment(best_state, cluster_size)


def solve_with_initial_solver(matrix: np.ndarray, cluster_size: int, init_solver, solver_factory):
    clusters = init_solver(matrix, cluster_size) if init_solver is not None else None
    return solve(matrix, cluster_size, solver_factory(init_state=clusters))


def annealing_solver_factory(max_attempts: int = 10, max_iters: int = 1000, init_state: List[Set[int]] = None):
    schedule = mlrose.ExpDecay()
    init_vec = clusters2assignment_vector(init_state) if init_state is not None else None
    solver = functools.partial(mlrose.simulated_annealing, schedule=schedule,
                               max_attempts=max_attempts, max_iters=max_iters, init_state=init_vec)
    return solver


def hill_climb_solver_factory(max_iters: int = 1000, init_state: List[Set[int]] = None):
    init_vec = clusters2assignment_vector(init_state) if init_state is not None else None
    solver = functools.partial(mlrose.hill_climb, max_iters=max_iters, init_state=init_vec)
    return solver


def random_hill_climb_solver_factory(max_iters: int = 10000, init_state: List[Set[int]] = None):
    init_vec = clusters2assignment_vector(init_state) if init_state is not None else None
    solver = functools.partial(mlrose.random_hill_climb, max_iters=max_iters, init_state=init_vec)
    return solver


def genetic_solver_factory(mutation_prob: int = 0.2, max_attempts: int = 1000):
    solver = functools.partial(mlrose.genetic_alg, mutation_prob=mutation_prob, max_attempts=max_attempts,
                               random_state=2)
    return solver

