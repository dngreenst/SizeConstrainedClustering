from typing import List

import dataclasses as dc
import numpy as np


def generate_scatter_based_matrix(agents_num: int,
                                  map_size: tuple,
                                  fractal_growth_probability: float,
                                  fractal_deviation: float,
                                  cost_function) -> np.array:
    agent_locations = _generate_agent_locations(agents_num=agents_num,
                                                map_size=map_size,
                                                fractal_growth_probability=fractal_growth_probability,
                                                fractal_deviation=fractal_deviation)

    return _create_mutual_concern_matrix_from_locations(agents_num=agents_num, agent_locations=agent_locations,
                                                        cost_function=cost_function)


def _generate_agent_locations(agents_num: int,
                              map_size: tuple,
                              fractal_growth_probability: float,
                              fractal_deviation: float) -> List[np.ndarray]:
    agent_locations = []
    for agent_index in range(agents_num):

        if agent_index == 0 or np.random.uniform(low=0.0, high=1.0) > fractal_growth_probability:
            # Uniform location
            agent_locations.append(_generate_location_uniformly(map_size=map_size))

        else:
            # fractal growth - choose an existing agent uniformly,
            # and randomize location according to gaussian distribution,
            # with deviation provided externally and mean being the chosen agent's location.

            focal_agent_location, focal_agent_index = _choose_agent_uniformly(agent_locations)
            map_location = _generate_location_with_normal_distribution(mean_location=focal_agent_location,
                                                                       deviation=fractal_deviation)
            agent_locations.insert(focal_agent_index, map_location)

    return agent_locations


def _generate_location_with_normal_distribution(mean_location: np.ndarray, deviation: float) -> np.ndarray:
    return np.random.normal(loc=mean_location, scale=deviation)


def _choose_agent_uniformly(agent_locations: List[np.ndarray]) -> (np.ndarray, int):
    current_agents_num = len(agent_locations)
    chosen_agent_index = np.random.randint(low=0, high=current_agents_num)
    return agent_locations[chosen_agent_index], chosen_agent_index


def _generate_location_uniformly(map_size: tuple) -> np.ndarray:
    return np.random.uniform(low=np.zeros(map_size.__len__()), high=np.array(map_size))


def _create_mutual_concern_matrix_from_locations(agents_num: int,
                                                 agent_locations: List[np.ndarray],
                                                 cost_function) -> np.ndarray:
    mutual_concern_matrix = np.zeros((agents_num, agents_num))
    for i in range(agents_num):
        for j in range(i):
            mutual_concern_matrix[i, j] = mutual_concern_matrix[j, i] = cost_function(agent_locations[i],
                                                                                      agent_locations[j])
    return mutual_concern_matrix


def negative_exponential_distance(location_i, location_j):
    return np.exp(-0.5*np.linalg.norm(location_i - location_j))


if __name__ == '__main__':
    mcm = generate_scatter_based_matrix(agents_num                  = 36,
                                        map_size                    = (50, 50),
                                        fractal_growth_probability  = 0.7,
                                        fractal_deviation           = 2.0,
                                        cost_function               = negative_exponential_distance)
    print('Done')
