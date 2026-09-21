import dill
import numpy as np
import torch
from scipy.linalg import expm
from scipy.signal import lfilter, ss2tf
from torch import nn
from torchdiffeq import odeint
from xitorch.interpolate import Interp1D


# TODO: Format comments to be consistent with PEP8.
class RCModel(nn.Module):
    """
    Custom Pytorch model for gradient optimization.
    Initialises with random parameters.

    scaling - Class containing methods to scale inputs from 0-1 back to their usual values and back again.
    transform - function to transform parameters e.g. sigmoid

    Runs using forward method should be sequential as the output is used as the initial condition for the next run,
    unless manually reset using self.iv
    """

    def __init__(self, building, scaling, Tout_continuous, transform=None, cooling_policy=None):

        super().__init__()
        self.building = building

        self.transform = transform  # transform performed on parameters e.g. sigmoid
        self.scaling = scaling  # InputScaling class (helper class to go between machine (0-1) and physical values)

        self.Tout_continuous = Tout_continuous  # Interp1D object

        self.cooling_policy = cooling_policy  # Neural net: pi(state) --> action
        self.action = 0  # initialise cooling action

        self.params = None  # initialised in initialise_parameters()
        self.loads = None  # initialised in initialise_parameters()
        self.initialise_parameters()  # initialise params and loads with random numbers

        self.ode_t = None  # Keeps track of t during integration. None is just to initialise attribute
        self.record_action = None  # records Q and t during integration
        self.cool_load = None  # cooling in Watts/m2
        self.gain_load = None  # gain in Watts/m2
        self.t0 = None  # unix epoch start time in seconds
        self.A = None  # System Matrix
        self.B = None  # Input Matrix
        self.iv = None  # initial value
        self.iv_array = None  # Interp1D object of pre found initial values. Means we can get correct iv with just time.

    def setup(self, dataset=None):
        """
        Setup must be called:
            - before the first forward pass.
            - after changing physical parameters (params, loads).
            - when changing the dataset i.e. for testing.
        """

        # Might not have changed from each forward pass but they're cheap enough.
        self._build_matrices()  # get A and B
        self._build_loads()  # get cool and gain load

        if dataset:
            self.iv_array = get_iv_array(self, dataset)

    # TODO: Allow for batches of data.
    def forward(self, t_eval, action=0):
        """
        Integrates the ode forward in time.

        building - Initialised RCModel Class Tout_continuous - A scipy interp1d function covering the whole of
        t_eval. Tout_continuous(t) = Outside temperature at time t iv - Initial value. Starting temperatures of all
        nodes t_eval - times function should return a solution. e.g. torch.arange(0, 10000, 30). ensure dtype=float32
        t0 - starting time if not 0
        """
        self.action = action
        self.record_action = []  # Keeps track of action at time during ODE integration.

        # check if t_eval is formatted correctly:
        if t_eval.dim() != 1:
            t_eval = t_eval.flatten()

        # t0 stores the starting epoch time and t_eval is array of seconds from start, [0, 1*dt, 2*dt, ...]
        self.t0 = t_eval[0]
        t_eval = t_eval - self.t0

        # THIS IS WRONG SINCE WE ARE NOW CALLING FORWARD MULTIPLE TIMES. IE for each time step.
        # Find the true iv from an initialised Inter1D object.
        # if self.iv_array:
        #     self.iv = self.iv_array(self.t0)

        # Format iv.
        self.iv = self.iv.reshape((2 + len(self.building.rooms), 1)).to(torch.float32)

        # integrate using fixed step (rk4) see torchdiffeq docs for more options.
        integrate = odeint(self.f_ode, self.iv, t_eval, method="rk4")  # https://github.com/rtqichen/torchdiffeq

        self.iv = None  # Causes error if iv is not reset before next forward pass.

        return integrate  # first two columns are external envelope nodes. i.e not rooms

    def f_ode(self, t, x):
        """
        Provides the function:
        dy/dx = Ax + Bu
        """
        # # get cooling action if policy is not None and 15 minutes has passed since last action
        # if self.cooling_policy:  # policy exists
        #     if t - self.ode_t >= 60*15:
        #         self.action, log_prob = self.cooling_policy.get_action(x[2:], t + self.t0)
        #         self.ode_t = t
        #
        #         if self.cooling_policy.training:  # if in training mode store log_prob
        #             self.cooling_policy.log_probs.append(log_prob)

        if self.cooling_policy:  # policy exists
            # record every time-step
            self.record_action.append([t, self.action])  # This is just used for plotting the cooling after.

        # Get energy input at timestep:
        Q_area = -self.cool_load * self.action  # W/m2
        Q_area = Q_area + self.gain_load  # add the constant gain term
        Q_watts = self.building.proportional_heating(Q_area)

        Tout = self.Tout_continuous(t.item() + self.t0)

        u = self.building.input_vector(Tout, Q_watts)

        return self.A @ x + self.B @ u

    def _build_matrices(self):
        """
        Build/re-build the A and B matrices with the current set of parameters.
        Keep track of parameters used, so we can check when to update.
        """
        theta, _ = self.get_physical_paramaters()

        # Produce matrix A and B from current parameters
        self.A = self.building.update_inputs(theta)
        self.B = self.building.input_matrix()

    def _build_loads(self):
        """
        Transform and scale loads.
        Keep track of loads used so we can only update when there is a difference.
        """
        _, loads = self.get_physical_paramaters()

        self.cool_load = loads[0, :]
        self.gain_load = loads[1, :]

    def initialise_parameters(self):
        params = torch.rand(self.building.n_params, dtype=torch.float32, requires_grad=True)
        loads = torch.rand((2, len(self.building.rooms)), dtype=torch.float32, requires_grad=True)

        # enables spread of initial parameters. Otherwise, sigmoid(rand) tends towards 0.5.
        if self.transform == torch.sigmoid:
            params = torch.logit(params)  # inverse sigmoid
            loads = torch.logit(loads)

        # make theta torch parameters
        self.params = nn.Parameter(params)

        # initialise the room cooling and gain loads
        self.loads = nn.Parameter(loads)

    def save(self, filename):
        """
        Parameters
        ----------
        filename : str
            Path to save the pickled RCModel to.
        Returns
        ----------
        filename : str
            Keeps format the same as RLLib
        """
        with open(filename, "wb") as dill_file:
            dill.dump(self, dill_file)

        return filename

    @staticmethod
    def load(filename):
        """
        Parameters
        ----------
        filename : str
            Path to the pickled RCModel to load.
        Returns
        ----------
        RCModel
        """
        with open(filename, "rb") as dill_file:
            return dill.load(dill_file)

    def get_physical_paramaters(self):
        """
        Go from params (most likely in the form: logit(rand(0,1)) to the true physical values.
        """
        # Transform parameters
        theta = self.transform(self.params) if self.transform else self.params

        # Scale inputs up to their physical values
        physically_scaled_params = self.scaling.physical_param_scaling(theta)

        # Transform loads
        loads = self.transform(self.loads) if self.transform else self.loads  # Watts/m2 for cooling and gain.

        physically_scaled_loads = self.scaling.physical_loads_scaling(loads)

        return physically_scaled_params, physically_scaled_loads


_UNIFORM_DT_TOL = 1e-6


def _foh_discretize(A, B, dt):
    """
    Exact discretization of dx/dt = A x + B u(t) over one step of length `dt`,
    assuming u(t) varies linearly between its values at the start and end of the
    step ("first-order hold" / ramp-invariant discretization). This matches the
    linear interpolation (Interp1D(method="linear")) already used for Tout/Tin
    elsewhere in this module, so the discrete recurrence this feeds reproduces
    what an ODE solver would compute for a piecewise-linear-input version of this
    system, exactly, without needing sub-steps.

    Returns Ad, Bd0, Bd1 such that x_{k+1} = Ad @ x_k + Bd0 @ u_k + Bd1 @ u_{k+1}.
    """
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + 2 * m, n + 2 * m))
    M[:n, :n] = A
    M[:n, n : n + m] = B
    M[n : n + m, n + m :] = np.eye(m)
    Md = expm(M * dt)

    Ad = Md[:n, :n]
    Bd_u = Md[:n, n : n + m]
    Bd_s = Md[:n, n + m :]

    Bd0 = Bd_u - Bd_s / dt
    Bd1 = Bd_s / dt
    return Ad, Bd0, Bd1


def _foh_discretize_batch(A, B, dt_arr):
    """
    Batched general-case counterpart to _foh_discretize for possibly non-uniform
    step sizes: computes a separate (Ad, Bd0, Bd1) for every entry in `dt_arr` in
    one vectorized call (torch.linalg.matrix_exp batches over the leading
    dimension), rather than calling scipy.linalg.expm once per step in a Python
    loop.
    """
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + 2 * m, n + 2 * m))
    M[:n, :n] = A
    M[:n, n : n + m] = B
    M[n : n + m, n + m :] = np.eye(m)

    M_t = torch.from_numpy(M)
    dt_t = torch.from_numpy(dt_arr)
    M_batch = M_t.unsqueeze(0) * dt_t.view(-1, 1, 1)
    Md_batch = torch.linalg.matrix_exp(M_batch)

    Ad = Md_batch[:, :n, :n]
    Bd_u = Md_batch[:, :n, n : n + m]
    Bd_s = Md_batch[:, :n, n + m :]

    Bd0 = Bd_u - Bd_s / dt_t.view(-1, 1, 1)
    Bd1 = Bd_s / dt_t.view(-1, 1, 1)
    return Ad.numpy(), Bd0.numpy(), Bd1.numpy()


def _integrate_latent_vectorized(Ad, Bd0, Bd1, W, x0):
    """
    Fully vectorized propagation of x_{k+1} = Ad @ x_k + Bd0 @ w_k + Bd1 @ w_{k+1}
    for a CONSTANT (Ad, Bd0, Bd1), i.e. uniform dt. Decomposes the response into a
    per-input-channel IIR filter (the exact digital-filter realization of this LTI
    system, run via scipy) plus the closed-form free response of x0 via
    eigendecomposition of Ad - avoids a Python loop over time steps entirely.

    W: (n_steps, 2) columns [Tout, Tin] at every point in t.
    x0: (2,) initial state.
    Returns: (n_steps, 2) state trajectory including x0 at index 0.
    """
    n_steps = len(W)
    n_states = Ad.shape[0]
    B_aug = np.concatenate([Bd0, Bd1], axis=1)  # (n_states, 4)

    # Bd1 uses the *next* sample's input, so it's driven by Tout/Tin shifted by
    # one index. The final entry of each shifted channel is never actually used
    # (the last output only depends on the first n_steps-1 inputs), so its value
    # is arbitrary - repeating the last sample is a convenient no-op filler.
    Tout, Tin = W[:, 0], W[:, 1]
    Tout_shift = np.concatenate([Tout[1:], Tout[-1:]])
    Tin_shift = np.concatenate([Tin[1:], Tin[-1:]])
    U = np.stack([Tout, Tin, Tout_shift, Tin_shift], axis=1)  # (n_steps, 4)

    C = np.eye(n_states)
    D = np.zeros((n_states, U.shape[1]))

    Z = np.zeros((n_steps, n_states))
    for j in range(U.shape[1]):
        num, den = ss2tf(Ad, B_aug, C, D, input=j)
        for i in range(n_states):
            Z[:, i] += lfilter(num[i], den, U[:, j])

    eigvals, eigvecs = np.linalg.eig(Ad)
    c = np.linalg.solve(eigvecs, x0)
    k = np.arange(n_steps)[:, None]
    free = ((eigvals**k) * c) @ eigvecs.T
    return Z + free.real


def _integrate_latent_loop_varying(Ad_batch, Bd0_batch, Bd1_batch, W, x0):
    """
    General-case counterpart to _integrate_latent_vectorized for non-uniform dt:
    (Ad, Bd0, Bd1) differ per step (see _foh_discretize_batch), so the recurrence
    can't be expressed as a single LTI filter and is instead applied step by step.
    Each step is a cheap 2x2 matrix-vector product (no interpolation), so this
    stays fast even though it isn't fully vectorized.
    """
    n_steps = len(W)
    X = np.empty((n_steps, len(x0)))
    X[0] = x0
    x = x0
    for k in range(n_steps - 1):
        x = Ad_batch[k] @ x + Bd0_batch[k] @ W[k] + Bd1_batch[k] @ W[k + 1]
        X[k + 1] = x
    return X


def _integrate_latent(A, B, t, W, x0):
    """
    Propagate the exact FOH discretization of dx/dt = A x + B u(t) across every
    point in `t`, given input samples `W` (one row per point in `t`) and initial
    state `x0`. Returns the state trajectory with the same length as `t`
    (including x0 at index 0).
    """
    dt_arr = np.diff(t)
    if np.allclose(dt_arr, dt_arr[0], rtol=_UNIFORM_DT_TOL, atol=_UNIFORM_DT_TOL):
        # Common case (matches production data, resampled to a fixed dt by
        # sort_data()): discretize once and integrate with a single vectorized
        # pass, no per-step Python loop.
        Ad, Bd0, Bd1 = _foh_discretize(A, B, dt_arr[0])
        return _integrate_latent_vectorized(Ad, Bd0, Bd1, W, x0)

    # dt varies between samples: discretize every step with its own dt in one
    # batched call rather than silently reusing a single-dt discretization that
    # would be wrong for the steps it doesn't match.
    Ad_batch, Bd0_batch, Bd1_batch = _foh_discretize_batch(A, B, dt_arr)
    return _integrate_latent_loop_varying(Ad_batch, Bd0_batch, Bd1_batch, W, x0)


def get_iv_array(model, dataset):
    """
    Estimate the initial values of the latent temperature nodes in the external walls, forcing outside and inside
    temperature to be from data. Only the latent nodes are free to change, meaning we can find their true values for
    a given model.

    Tout --R--T1--R--T2--R-- Tin
              |      |
              C      C

    Tout and Tin are exogenous inputs (not functions of T1/T2), and Q never enters this reduced system - so this is a
    linear time-invariant state-space system with known input trajectories, which is discretized exactly (see
    _foh_discretize) rather than numerically integrated with an ODE solver.
    """

    with torch.no_grad():
        t_eval, temp_data = dataset.get_all_data()
        Tin_continuous = Interp1D(t_eval, temp_data[:, 0 : len(model.building.rooms)].T, method="linear")

        bl = model.building

        # Recalculate the A & B matrices. We could chop and re jig from the full matrices, but it is not super
        # simple so recalculating is less risky.
        A = torch.zeros([2, 2])
        A[0, 0] = bl.surf_area * (-1 / (bl.Re[0] * bl.Ce[0]) - 1 / (bl.Re[1] * bl.Ce[0]))
        A[0, 1] = bl.surf_area / (bl.Re[1] * bl.Ce[0])
        A[1, 0] = bl.surf_area / (bl.Re[1] * bl.Ce[1])
        A[1, 1] = bl.surf_area * (-1 / (bl.Re[1] * bl.Ce[1]) - 1 / (bl.Re[2] * bl.Ce[1]))

        B = torch.zeros([2, 2])
        B[0, 0] = bl.surf_area / (bl.Re[0] * bl.Ce[0])
        B[1, 1] = bl.surf_area / (bl.Re[2] * bl.Ce[1])

        # check if t_eval is formatted correctly:
        if t_eval.dim() > 1:
            t_eval = t_eval.squeeze(0)

        avg_tout = model.Tout_continuous(t_eval).mean()
        avg_tin = Tin_continuous(t_eval).mean()

        model.iv = steady_state_iv(model, avg_tout, avg_tin)  # Use avg temp as a good starting guess for iv.

        # Inputs are only needed at the dataset's own sample points: Tin is just the
        # raw (already-loaded) room data - no interpolation needed at all - and Tout
        # needs exactly one batched Interp1D call for the whole trajectory, not one
        # per ODE sub-step. t_eval is used directly (absolute time, matching how
        # avg_tout/avg_tin above and model.Tout_continuous's own domain are defined) -
        # there's no need to shift to a relative time origin here, unlike the old
        # torchdiffeq-based implementation, which needed small relative values to
        # avoid float32 precision loss inside torchdiffeq's internal dtype cast.
        external_rooms = bl.connectivity_matrix[0, 1:]
        # Tin is the average temperature of spaces connected to an external wall.
        Tin_agg = (temp_data[:, 0 : len(bl.rooms)] * external_rooms).mean(dim=1)
        Tout_vals = model.Tout_continuous(t_eval)

        W = np.stack(
            [Tout_vals.numpy().astype(np.float64), Tin_agg.numpy().astype(np.float64)],
            axis=1,
        )
        x0 = model.iv[0:2].reshape(-1).numpy().astype(np.float64)
        X = _integrate_latent(A.numpy().astype(np.float64), B.numpy().astype(np.float64), t_eval.numpy(), W, x0)

        integrate = torch.tensor(X, dtype=torch.float32)

        # Add on inside temperature data to be used to initialise rooms at the correct temp.
        iv_array = torch.empty(len(integrate), len(bl.rooms) + 2)
        iv_array[:, 0:2] = integrate
        iv_array[:, 2:] = Tin_continuous(t_eval).T
        iv_array = Interp1D(t_eval, iv_array.T, method="linear")

    return iv_array


def steady_state_iv(model, temp_out, temp_in):
    """
    Calculate the initial conditions of the latent variables given a steady state indoor and outdoor temperature.
    Initial values of room nodes are set to temp in.

    temp_out: float
        Steady state outside temperature.
    temp_in: tensor
        Steady state inside temperature. len(temp_in) = n_rooms
    :return: tensor
        Column tensor of initial values at each node.
    """

    I = (temp_out - temp_in) / sum(model.building.Re)  # I=V/R  # noqa: E741
    v1 = temp_out - I * model.building.Re[0]
    v2 = v1 - I * model.building.Re[1]

    iv = torch.tensor([[v1], [v2], [temp_in]], dtype=torch.float32)

    return iv
