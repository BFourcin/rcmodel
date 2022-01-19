import numpy as np
import torch
from xitorch.interpolate import Interp1D


class Building:
    """
    A class containing all the rooms of a building (2D)
    """

    def __init__(self, rooms, height=1):

        self.rooms = rooms  # list of Room classes within the building
        self.height = height  # height of all Rooms in the building

        #  Parameters to be properly initialised later in self.update_inputs()
        self.Re = [0, 0, 0]      # external resistance
        self.Ce = [0, 0]      # external capacitance
        self.Rint = 0    # internal wall resistance
        self.gain = 0    # constant heat gain (W/m^2)

        self.Walls = self.sort_walls()


        # get total external area:
        surf_area = 0
        for i in range(len(self.Walls)):
            if self.Walls[i].is_external:
                surf_area += self.Walls[i].area

        self.surf_area = surf_area
        self.n_params = self.get_n_params()  # Number of parameters needed for model i.e len(theta)

    def make_connectivity_matrix(self):
        """
        Creates connectivity matrix of the thermal conductivity (W/K) of rooms in building.
        The first row/col is an external connection.
        Thermal conductivity is calculated by K = area/resistance


        Output matrix is in the form:
            [External TA TB ... Tn]
        [External]
        [TA]
        [TB]
        .
        .
        [Tn]

        """

        n = len(self.rooms)

        knect = torch.zeros([n, n])

        # Iterate through each room and check if room shares any walls with other rooms
        for rm_x in range(0, n - 1):
            for rm_y in range(rm_x + 1, n):

                # compare first wall in rm_x with all walls in rm_y then move on to next wall
                for wl_x in range(len(self.rooms[rm_x].walls)):
                    for wl_y in range(len(self.rooms[rm_y].walls)):

                        if self.rooms[rm_x].walls[wl_x] == self.rooms[rm_y].walls[wl_y]:
                            area = self.Walls[self.rooms[rm_x].walls[wl_x]].area
                            resistance = self.Walls[self.rooms[rm_x].walls[wl_x]].resistance

                            # Sum the area to find total shared area to a room
                            knect[rm_x, rm_y] = knect[rm_x, rm_y] + area / resistance
                            knect[rm_y, rm_x] = knect[rm_x, rm_y]  # matrix is symmetric

        # Now rooms are checked if they are external and stored in a vector.
        is_ex = torch.zeros(n)
        for rm in range(n):
            for wl in range(len(self.rooms[rm].walls)):
                wl_class = self.Walls[self.rooms[rm].walls[wl]]

                if wl_class.is_external:
                    is_ex[rm] = is_ex[rm] + wl_class.area / wl_class.resistance

        # The is_external vector is used as the first row/col for the connection matrix
        knect = torch.vstack((is_ex, knect))
        is_ex = torch.cat(
            (torch.tensor([0]), is_ex))  # adds 0 to start of array (account for external to external connection)

        is_ex = is_ex.unsqueeze(0)  # adds dimension
        knect = torch.hstack((is_ex.T, knect))

        return knect

    def make_system_matrix(self):
        """
        Creates the system/state matrix (A) for the Building.
        This will be used in the State Space equation: xdot = A@x + B@u

        The system matrix (A) is in the form:

        x = [Te1, Te2, TA , TB , ... Tn ]

        A = [A00, A01, A02, A03, ... A0n], #dTe1/dt
            [A10, A11, A12, A13, ... A1n], #dTe2/dt
            [A20, A21, A22, A23, ... A2n], #dTA/dt
            .
            .
            .

        """

        knect = self.make_connectivity_matrix()

        n = len(knect[0])

        off = len(self.Ce) - 1  # Num nodes not seen in connection matrix which need to be offset in A.

        # Plus "off" accounts for the node Te1 in the external envelope which is not part of the connection matrix
        A = torch.zeros([n + off, n + off])

        # Hard code in the external to external connections. These will always be the same.
        A[0, 0] = self.surf_area * (-1 / (self.Re[0] * self.Ce[0]) - 1 / (self.Re[1] * self.Ce[0]))
        A[0, 1] = self.surf_area / (self.Re[1] * self.Ce[0])
        A[1, 0] = self.surf_area / (self.Re[1] * self.Ce[1])
        A[1, 1] = -self.surf_area / (self.Re[1] * self.Ce[1])

        for row in range(n):
            if row == 0:
                c = self.Ce[-1]
            else:
                c = self.rooms[row - 1].capacitance  # Heat Capacity (J/K)

            for col in range(n):
                K = knect[row, col]  # Thermal Conductance (W/K) K = A(m^2)/R(K.m^2/W)

                A[row + off, col + off] = A[row + off, col + off] + K / c
                A[row + off, row + off] = A[row + off, row + off] - K / c

        return A

    def sort_walls(self):
        """
        Function which combines 3 separate functions in the correct order.
        All unique walls in the building are represented in a list of Wall class instances.
        Walls previously defined in the Room class are also updated to be an index of this list.
        """

        Walls = self.get_walls()
        self.update_rooms(Walls)
        Walls = self.get_external_walls(Walls)

        return Walls

    def get_walls(self):
        """
        Looks at every wall in every room within the building and produces a list of unique walls.
        This list is then used to create a Wall class for each unique wall.
        Output is list all Wall classes. [Wall1, Wall2, ... Walln]
        """
        # First get list of all walls in all rooms.
        walls = []
        for rm in range(len(self.rooms)):
            for wl in range(len(self.rooms[rm].walls)):
                walls.append(self.rooms[rm].walls[wl])

        # Now the duplicates need to be deleted so only unique walls remain.
        duplicate_wls = []
        for wl1 in range(len(walls) - 1):
            for wl2 in range(wl1 + 1, len(walls)):
                if set(walls[wl1]) == set(walls[wl2]):
                    duplicate_wls.append(wl1)  # A list of indexes is created

        # Delete duplicates from the original list using the indexes found above
        for index in sorted(duplicate_wls, reverse=True):
            del walls[index]

        # A Wall class is instantiated for each unique wall and saved in the list Walls
        Walls = []
        for wl in walls:
            Walls.append(self.Wall(wl, self.Rint, self.height))

        return Walls

    def update_rooms(self, Walls):
        """
        Update the Room.walls variable to be an index matching the list Walls which contains the class instances of each unique wall.
        """
        # We can now use this list, Walls, as an index for the Room classes.
        for rm in range(len(self.rooms)):
            for wl in range(len(self.rooms[rm].walls)):
                for Walls_indx in range(len(Walls)):

                    # For each wall in the room class replace with an index to the matching wall in the list Walls
                    if set(self.rooms[rm].walls[wl]) == set(Walls[Walls_indx].coordinates):
                        self.rooms[rm].walls[wl] = Walls_indx
                        break

                    # Check for no matches
                    elif Walls_indx == len(Walls):
                        print("Error: Room: ", rm, ", Wall: ", wl, ". Was not matched")

    def get_external_walls(self, Walls):
        """
        Finds all external walls and updates the boolean variable is_external in the Wall Class.
        """

        # A wall is external if it is not shared by another room.
        rm_wls = []
        for rm in range(len(self.rooms)):
            for wl in range(len(self.rooms[rm].walls)):
                rm_wls.append(
                    self.rooms[rm].walls[wl])  # Creates list of walls in each room. Can now check for multiples.

        for i in rm_wls:
            occurrences = torch.count_nonzero(torch.tensor(rm_wls) == i)

            # Occurance of 1 means the wall is only seen in a single room meaning it must be an external wall.
            if occurrences == 1:
                Walls[i].is_external = True
                Walls[i].resistance = self.Re[2]  # Change resistance of wall

        return Walls

    def input_vector(self, Tout, Q):
        """
        Input vector in form:
        u = [Tout, QA, QB, ... Qn]
        """
        # Check and make 2d matrix if 1d.
        # add_dim = lambda x: torch.tensor([x]) if x.ndim == 1 else x

        # Q = np.array(Q)
        # Q = add_dim(Q)
        # Q = torch.tensor(Q, dtype=torch.float32).unsqueeze(0) #add dim
        Q = Q.unsqueeze(0)  # add dim

        Tout = Tout.type(torch.float32)
        Tout = torch.reshape(Tout, (1, 1))

        if len(Q[0]) == len(self.rooms):
            u = torch.cat((Tout, Q), dim=1).T

            return u

        else:
            raise ValueError("input_vector: Q needs to have 1 column per room")

    def input_matrix(self):
        """
        Produces the input matrix: B
        """
        # u = [[Tout, QA, QB,...]].T

        off = len(self.Ce)  # offset needed for external nodes
        num_inputs = len(self.rooms) + 1

        B = torch.zeros((off + len(self.rooms), num_inputs))

        # Set Tout input:
        B[0, 0] = self.surf_area / (self.Re[0] * self.Ce[0])

        # Set the Q inputs for each room
        for rm in range(len(self.rooms)):
            B[rm + off, rm + 1] = 1 / self.rooms[rm].capacitance

        return B

    def update_wall(self, wall_ID, resistance=-1):
        """
        Used to update the resistance of the wall.
        Wall is chosen by either supplying the coordinates of the wall or the index number to the Walls list.
        Coordinates must be supplied in a tuple e.g. ((x1, y1), (x2, y2))
        Function returns its wall index number in the list Walls
        """
        is_coord = False

        # check if ID is a coordinate or an index
        if isinstance(wall_ID, int):
            wl_indx = wall_ID
        elif len(self.Walls[0].coordinates) == len(wall_ID):
            is_coord = True
        else:
            return print("wall_ID has been input incorrectly")

        if is_coord:
            for wl in range(len(self.Walls)):

                if set(self.Walls[wl].coordinates) == set(tuple(wall_ID)):
                    wl_indx = wl
                    break
        try:
            if resistance != -1:
                self.Walls[wl_indx].resistance = resistance


        except:
            print("Could not find the wall")

        return wl_indx

    def plt_ext_walls(self):
        """
        Plots all walls and highlights in red the external walls. Used to check if model has correctly identified
        """
        from matplotlib import pyplot as plt
        import numpy as np

        for wl in range(len(self.Walls)):
            c = np.array(self.Walls[wl].coordinates)
            if self.Walls[wl].is_external:
                ex, = plt.plot(c[:, 0], c[:, 1], 'r', label='External Walls')
            else:
                int, = plt.plot(c[:, 0], c[:, 1], 'k', label='Internal Walls')

        plt.legend([ex, int], ['External Walls', 'Internal Walls'])

        return plt.show()

    def categorise_theta(self, theta):
        """
        Function to split theta into its different categories.

        theta is a 1D vector with categories in the following order:
        theta = [room capacitance, external capacitance, external resistance, internal wall resistance, gain]
        """

        indx = 1
        rm_cap = theta[0:indx]
        indx += 2
        ex_cap = theta[indx - 2:indx]
        indx += 3
        ex_r = theta[indx - 3:indx]
        indx += 1
        wl_r = theta[indx - 1]
        indx += 1
        gain = theta[indx - 1]

        return rm_cap, ex_cap, ex_r, wl_r, gain

    def update_inputs(self, theta):
        """
        Function to update the instance with new variables, theta. Must be structured correctly.
        Note: system matrices will have to be reproduced e.g. A, B

        theta is a 1D vector with categories in the following order:
        theta = [room capacitance, external capacitance, external resistance, each wall resistance, gain]


        returns Q_avg: The mean W/m^2 for the building
        """

        rm_cap, ex_cap, ex_r, wl_r, gain = self.categorise_theta(theta)

        # update room capacitance
        for i in range(len(self.rooms)):
            self.rooms[i].capacitance = rm_cap[0] * self.rooms[i].area

        # update external capacitance
        self.Ce = ex_cap

        # update external resistance
        self.Re = ex_r

        # update wall resistance
        self.Rint = wl_r
        # This code is not great, need way of assigning wall resistance individually in the future
        for i in range(len(self.Walls)):
            if self.Walls[i].is_external:
                self.Walls[i].resistance = ex_r[2]
            else:
                self.Walls[i].resistance = wl_r


        # update gain
        self.gain = gain

        A = self.make_system_matrix()

        return A

    def get_n_params(self):
        """Find how many parameters the model needs by testing each number."""
        n_params = 1
        while True:
            params = torch.ones(n_params)
            try:
                self.categorise_theta(params)
            except:
                n_params += 1

            else:
                break  # This is reached if no error occurs

        return n_params

    # Heating:
    def proportional_heating(self, Q_avg):
        """
        Input: Q_avg (W/m^2)
        Calc: Q_avg * rm_area
        Outputs: Q (Watts)
        """

        Q = torch.zeros(len(self.rooms))
        for i in range(len(self.rooms)):
            Q[i] = Q_avg * self.rooms[i].area

        return Q

    def time_angle(self, circ):
        """outputs angle from x axis to point on circle. Used to represent time in a day"""

        theta = np.angle(circ) * 180 / np.pi
        # add 360 if y axis is negative.
        theta = np.where(circ.imag < 0, theta + 360, theta)

        return torch.tensor(theta)

    def Q_control(self, t, bound_A, bound_B):
        """
        Produces a step function for given timeseries, turn on when time of day is between the thresholds A-B.
        t is unix epoch time (seconds since 00:00 Jan 1970), t is converted to an angle (degrees 0-360) repeating every 24hrs.
        Theta_A and Theta_B represent the start and stop day angle.


        t - time evaluations, unix epoch time
        bound_* - enter value between -6:6 this is then scaled up to between 0 and 360 using a sigmoid function.

        Output: step function off (0) or on (1) for each point of the timeseries. Each row is a different room.
                tensor([[rm1],
                        [rm2],
                        ....
                        [rmn]])

        """

        s_per_day = 24 * 60 ** 2

        # function which bounds value between 0:360
        bound = lambda x: 360 * torch.sigmoid(x)

        # scale theta between 0:360 degrees
        theta_A = bound(bound_A)
        theta_B = bound(bound_B)

        # get angle (i.e time of day) for timeseries
        circ = np.e ** (1j * 2 * np.pi / s_per_day * t)
        time_degree = self.time_angle(circ.cpu())

        # Control logic:
        # turn Q on if time_degree is between thresholds theta_A and theta_B.
        # Do this for each room. num rooms = len(theta_A)

        Q_on_off = torch.zeros((len(theta_A), len(t)))
        stretch = 2e1  # the larger this number the steeper the step function is. But you lose the ability to autograd the gradient.
        # 2e1 means the step goes from 0-1 in roughly two minutes and grad is calculated fine.
        for i in range(len(theta_A)):
            if theta_A[i] < theta_B[i]:
                condition1 = torch.sigmoid((time_degree - theta_A[i]) * stretch)  # True if time>A
                condition2 = torch.sigmoid((theta_B[i] - time_degree) * stretch)  # True if time<B
                Q_on_off[i] = torch.sigmoid(
                    (condition1 + condition2 - 1.9) * stretch)  # True if con1 & con2 on between A-B
            else:
                condition1 = torch.sigmoid((time_degree - theta_A[i]) * stretch)
                condition2 = torch.sigmoid((theta_B[i] - time_degree) * stretch)
                Q_on_off[i] = torch.sigmoid((condition1 + condition2 - 0.4) * stretch)  # on between B-A

        return Q_on_off

    def Q_continuous(self, t, Q_avg, theta_A1, theta_B1):

        Q_on_off = self.Q_control(t, theta_A1, theta_B1)  # get control step funciton for each room

        Q_on_off = Q_on_off

        Q = Q_on_off * Q_avg.unsqueeze(1)  # multiply by Q_avg for each room

        Q_cont = Interp1D(t, Q, method='linear')

        return Q_cont

    class Wall:
        """
        A nested class, this is used to contain unique information about each wall in the building.
        """

        def __init__(self, coordinates, resistance, height):
            self.coordinates = coordinates
            self.resistance = resistance
            self.is_external = False  # This is updated later
            self.height = height
            self.area = self.get_area()

        def get_area(self):
            wl = torch.tensor(self.coordinates)  # put wall in array for easy calc of area
            length = torch.linalg.norm(wl[0] - wl[1], dtype=torch.float)

            return length * self.height
