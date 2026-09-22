import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull


class Room:
    """
    A room in a building (2D).
    Coordinates are in METRES and must be entered in a clockwise or anti-clockwise order.
    """

    def __init__(self, name, coordinates):

        self.name = name
        self.coordinates = torch.tensor(coordinates)
        self.capacitance = 0  # Dummy Value, is changed during optimisation process.

        self.hull = ConvexHull(coordinates)
        # For a 2D ConvexHull scipy names these the other way round to what you would expect:
        # .volume is the enclosed area (m^2) and .area is the perimeter (m). This used to read
        # .area, which silently made C_rm, cool and gain per-metre-of-perimeter rather than
        # per-m^2 - for the LSI seminar room, 37.3 in place of 85.3.
        self.area = self.hull.volume  # floor area (m^2)
        self._path = mpltPath.Path(self.coordinates)

        # Iterate through sorted vertices and store each pair which form a wall
        walls = []
        for i in range(len(self.coordinates) - 1):
            walls.append((tuple(coordinates[i]), tuple(coordinates[i + 1])))
        walls.append((tuple(coordinates[-1]), tuple(coordinates[0])))
        self.walls = walls  # list of walls in room

    def pltrm(self, ax=None, colour=None):

        c = self.coordinates

        if ax is None:
            ax = plt

        line = ax.plot(
            np.hstack([c[:, 0], c[0, 0]]),
            np.hstack([c[:, 1], c[0, 1]]),
            c=colour,
        )
        return line

    def points_in_room(self, P):
        """Boolean answering if point is inside the polygon.

        P is in form:
                P = [[x1,y1],[x2,y2]...]
        """
        return self._path.contains_points(P)
