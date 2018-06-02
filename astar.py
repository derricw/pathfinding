import math
import numpy as np
from itertools import product

import matplotlib.pyplot as plt


class Node(object):
    def __init__(self, pos):
        self.pos = pos
        self.neighbors = []
        self.visited = False
        self.distance = None
        self.previous = None

    def is_neighbor(self, node):
        nx, ny = node.pos
        x, y = self.pos
        return (nx - 1 <= x <= nx + 1) and (ny - 1 <= y <= ny + 1)

    def __repr__(self):
        return "({}, {})".format(*self.pos)


class Graph(object):
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        for other_node in self.nodes:
            if other_node.is_neighbor(node):
                node.neighbors.append(other_node)
                other_node.neighbors.append(node)
        self.nodes.append(node)

    def add_point(self, pos):
        self.add_node(Node(pos))


def dijkstras(graph, initial, destination):
    # find the best route from initial to destination
    unvisited = [node for node in graph.nodes]
    for node in unvisited:
        node.distance = 1E100 #effectivly infinite
    initial.distance = 0.0

    current_node = initial

    while current_node is not destination:
        print(" -- {} --".format(current_node))
        # find tentative distances for neighbors
        for n in current_node.neighbors:
            d = math.sqrt((n.pos[0]-current_node.pos[0])**2 + (n.pos[1]-current_node.pos[1])**2)
            tentative_distance = current_node.distance + d
            if tentative_distance < n.distance:
                n.distance = tentative_distance
                n.previous = current_node
                print("{} -> {}".format(n, tentative_distance))

        # remove current node
        unvisited.remove(current_node)
        if unvisited:
            # choose next node
            next_node = None
            closest_distance = 1e100
            for node in unvisited:
                if node.distance < closest_distance:
                    next_node = node
                    closest_distance = node.distance

        current_node = next_node

    # did we make it to the destination?
    if not destination.previous:
        raise Exception("We failed to find a path to the destination.")
    
    # get optimal path by traversing the optimal nodes from destination
    optimal_path = [destination]
    
    optimal = destination.previous
    while optimal is not None:
        optimal_path.append(optimal)
        optimal = optimal.previous

    return destination.distance, optimal_path[::-1]


def astar(graph, initial, destination):
    pass



def main():
    x_range = 10
    y_range = 10

    x = np.arange(0, x_range)
    y = np.arange(0, y_range)

    node_positions = list(product(x, y))

    graph = Graph()
    for n in node_positions:
        graph.add_point(n)

    initial = graph.nodes[0]
    destination = graph.nodes[-1]

    print(dijkstras(graph, initial=initial, destination=destination))

    #x = [node.pos[0] for node in graph.nodes]
    #y = [node.pos[1] for node in graph.nodes]
    #plt.scatter(x,y)
    #plt.show()

if __name__ == '__main__':
    main()
