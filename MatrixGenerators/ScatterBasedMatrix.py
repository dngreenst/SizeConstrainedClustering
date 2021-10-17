from typing import List

import dataclasses as dc
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt


@dc.dataclass
class MapShape:
    horizontal: float
    vertical: float


@dc.dataclass
class MapLocation:
    horizontal: float
    vertical: float

    def to_numpy(self):
        return np.array((self.horizontal, self.vertical))


def generate_scatter_based_matrix(agents_num: int,
                                  map_size: MapShape,
                                  fractal_growth_probability: float,
                                  fractal_deviation: float,
                                  cost_function) -> np.array:
    agent_locations = _generate_agent_locations(agents_num=agents_num,
                                                map_size=map_size,
                                                fractal_growth_probability=fractal_growth_probability,
                                                fractal_deviation=fractal_deviation)

    return _create_mutual_concern_matrix_from_locations(agents_num=agents_num, agent_locations=agent_locations,
                                                        map_size=map_size, cost_function=cost_function)


def _generate_agent_locations(agents_num: int,
                              map_size: MapShape,
                              fractal_growth_probability: float,
                              fractal_deviation: float) -> List[MapLocation]:
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


def _generate_location_with_normal_distribution(mean_location: MapLocation, deviation: float) -> MapLocation:
    numpy_agent_location = \
        np.random.normal(loc=np.array([mean_location.horizontal, mean_location.vertical]),
                         scale=deviation)
    return MapLocation(horizontal=numpy_agent_location[0], vertical=numpy_agent_location[1])


def _choose_agent_uniformly(agent_locations: List[MapLocation]) -> MapLocation:
    current_agents_num = len(agent_locations)
    chosen_agent_index = np.random.randint(low=0, high=current_agents_num)
    return agent_locations[chosen_agent_index], chosen_agent_index


def _generate_location_uniformly(map_size: MapShape) -> MapLocation:
    numpy_agent_location = np.random.uniform(low=np.array([0.0, 0.0]),
                                             high=np.array([map_size.horizontal, map_size.vertical]))
    return MapLocation(horizontal=numpy_agent_location[0], vertical=numpy_agent_location[1])


def _create_mutual_concern_matrix_from_locations(agents_num: int,
                                                 agent_locations: List[MapLocation],
                                                 map_size: MapShape,
                                                 cost_function) -> np.array:

    # # DEBUG (locations scatter map visualization)
    # locations_matrix = np.zeros((int(map_size.horizontal), int(map_size.vertical)))
    # for i in range(len(agent_locations)):
    #     locations_matrix[int(np.floor(agent_locations[i].horizontal))][int(np.floor(agent_locations[i].vertical))] = 1
    # ax = sns.heatmap(locations_matrix, linewidth=0.1)
    # plt.show()

    mutual_concern_matrix = np.zeros((agents_num, agents_num))

    for i in range(agents_num):
        for j in range(i):
            mutual_concern_matrix[i, j] = mutual_concern_matrix[j, i] = cost_function(agent_locations[i],
                                                                                      agent_locations[j],
                                                                                      map_size)

    return mutual_concern_matrix


def normalized_euclidean_distance(coord_i, coord_j, map_size):
    power = 8
    max_distance = np.sqrt(map_size.horizontal**2 + map_size.vertical**2)
    return ((max_distance - np.linalg.norm(coord_i.to_numpy() - coord_j.to_numpy())) / max_distance)**power


def negative_exponential_distance(coord_i, coord_j, map_size):
    return np.exp(-0.5*np.linalg.norm(coord_i.to_numpy() - coord_j.to_numpy()))


if __name__ == '__main__':
    mcm = generate_scatter_based_matrix(agents_num                  = 36,
                                        map_size                    = MapShape(50.0, 50.0),
                                        fractal_growth_probability  = 0.7,
                                        fractal_deviation           = 2.0,
                                        cost_function               = negative_exponential_distance)
    print('Done')
