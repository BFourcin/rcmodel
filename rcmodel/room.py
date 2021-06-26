import torch

class Room:

    def __init__(self,name,capacitance,coordinates):
        """
        A room in a building (2D).
        Coordinates must be entered in a clockwise or anti-clockwise order.
        """
        self.name = name
        self.capacitance = capacitance
        self.coordinates = torch.tensor(coordinates)
        self.area = self.polyarea(self.coordinates)

        # Iterate through sorted vertices and store each pair which forms a wall
        walls = []
        for i in range(len(self.coordinates)-1):
            walls.append((tuple(coordinates[i]), tuple(coordinates[i+1])))
        walls.append((tuple(coordinates[-1]), tuple(coordinates[0])))
        self.walls = walls #list of walls in room


    def pltrm(self):
        from matplotlib import pyplot as plt
        import numpy as np
        c = self.coordinates
        return plt.plot(np.append(c[:,0], c[0][0]) , np.append(c[:,1], c[0][1]))

    def polyarea(self, coordinates):
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coordinates)

        return hull.area
