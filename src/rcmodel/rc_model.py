import dill
import numpy as np
import torch
from scipy.linalg import expm
from scipy.signal import lfilter, ss2tf
from torch import nn
from torchdiffeq import odeint
from xitorch.interpolate import Interp1D

# Single source of truth for the names and ORDER of the model's free parameters.
# PARAM_KEYS must match Building.categorise_theta(); LOAD_KEYS must match the row
# order of InputScaling.energy_param_range (row 0 cool, row 1 gain, row 2 solar).
#   cool  - cooling limit, W/m2 of floor area, switched by the action.
#   gain  - constant heat gain, W/m2 of floor area.
#   solar - dimensionless fraction p of global horizontal irradiance reaching the room:
#           solar gain (W) = p * GHI(t) (W/m2) * floor area (m2).
PARAM_KEYS = ("C_rm", "C1", "C2", "R1", "R2", "R3", "Rin")
LOAD_KEYS = ("cool", "gain", "solar")
RC_PARAM_KEYS = PARAM_KEYS + LOAD_KEYS


def scaled_params_to_tensors(values, n_rooms):
    """
    Turn a {name: value} mapping of machine-space parameters into the (params, loads)
    tensors RCModel stores.

    Parameters
    ----------
    values : dict
        Keys must cover RC_PARAM_KEYS. Every value is in machine space - the linear map
        taking each parameter's configured [min, max] range to [0, 1] - NOT physical units;
        use pbt.physical_to_scaled to get there. Values outside [0, 1] are allowed: they
        are physical values outside the configured range, which a PBT perturbation
        legitimately produces. RCModel.set_parameters() checks the physical values are
        valid. The LOAD_KEYS ("cool", "gain", "solar") accept either a single value used for
        every room, or one value per room.
    n_rooms : int
        Number of rooms in the building.

    Returns
    -------
    (params, loads) : (torch.Tensor, torch.Tensor)
        Shapes (len(PARAM_KEYS),) and (len(LOAD_KEYS), n_rooms).
    """
    missing = [key for key in RC_PARAM_KEYS if key not in values]
    if missing:
        raise KeyError(f"Missing RC parameters: {missing}")

    def as_scalar(value):
        return float(np.asarray(value).item())

    def as_room_vector(value, name):
        vector = torch.as_tensor(np.asarray(value, dtype=np.float32)).flatten().to(torch.float32)
        if vector.numel() == 1:
            vector = vector.repeat(n_rooms)
        if vector.numel() != n_rooms:
            raise ValueError(
                f"Each room needs a '{name}' load - give one value per room, or a single value for all. "
                f"Got {vector.numel()} values for {n_rooms} rooms."
            )
        return vector

    params = torch.tensor([as_scalar(values[key]) for key in PARAM_KEYS], dtype=torch.float32)
    loads = torch.stack([as_room_vector(values[key], key) for key in LOAD_KEYS])

    for name, tensor in (("params", params), ("loads", loads)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Scaled {name} must be finite, got {tensor.tolist()}.")

    return params, loads


# TODO: Format comments to be consistent with PEP8.
class RCModel(nn.Module):
    """
    3R2C thermal model of a building.

    Each room's heat input is Q(t) = floor area * (gain - cool * action + solar * GHI(t)), in W.

    scaling - Class containing methods to scale inputs from 0-1 back to their usual values and back again.
    transform - optional function applied to parameters before scaling, e.g. sigmoid. Leave as None
        (the default) to hold parameters directly in 0-1 machine space, which is what the PBT search
        expects - see set_parameters(). The sigmoid/logit parameterisation only existed to keep
        gradient descent on unbounded parameters well behaved, and there is no gradient descent here.

    Parameters are NOT optimised by gradient descent: they are supplied from outside (a PBT trial's
    config) and held fixed while the cooling policy trains against them. Nothing in this class builds
    an autograd graph.

    Runs using forward method should be sequential as the output is used as the initial condition for the next run,
    unless manually reset using self.iv

    ghi_continuous - optional callable (normally an Interp1D) giving global horizontal irradiance in W/m2 at
        absolute (unix epoch) times, like Tout_continuous. None means no solar data: GHI is taken as zero, so
        the solar parameter has no effect.
    """

    def __init__(self, building, scaling, Tout_continuous, transform=None, cooling_policy=None, ghi_continuous=None):

        super().__init__()
        self.building = building

        self.transform = transform  # transform performed on parameters e.g. sigmoid
        self.scaling = scaling  # InputScaling class (helper class to go between machine (0-1) and physical values)

        self.Tout_continuous = Tout_continuous  # Interp1D object
        self.ghi_continuous = ghi_continuous  # Interp1D object, or None for no solar data

        self.cooling_policy = cooling_policy  # Kept for plotting; the action comes from the environment.
        self.action = 0  # initialise cooling action

        self.params = None  # initialised in initialise_parameters()
        self.loads = None  # initialised in initialise_parameters()
        self.initialise_parameters()  # initialise params and loads with random numbers

        self.ode_t = None  # Keeps track of t during integration. None is just to initialise attribute
        self.record_action = None  # records Q and t during integration
        self.cool_load = None  # cooling in Watts/m2
        self.gain_load = None  # gain in Watts/m2
        self.solar_load = None  # solar fraction p of GHI (dimensionless)
        self.t0 = None  # unix epoch start time in seconds
        self.A = None  # System Matrix
        self.B = None  # Input Matrix
        self.iv = None  # initial value
        self.iv_array = None  # Interp1D object of pre found initial values. Means we can get correct iv with just time.

        # Cache of exact discretisations keyed by timestep, populated lazily by _get_discretisation()
        # and thrown away by setup() whenever A/B change. Not part of state_dict - it is derived data.
        self._disc_cache = {}

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

        # A and B have just been rebuilt, so any cached discretisation of them is stale.
        self._disc_cache = {}

        if dataset:
            self.iv_array = get_iv_array(self, dataset)

    def set_parameters(self, values):
        """
        Replace the model's free parameters with a new set, in machine space.

        This is the entry point PBT uses: a trial's parameters live in its Tune config and
        arrive here. Call setup() afterwards to rebuild A/B and recompute iv_array - the
        environment's update_from_config() does that for you.

        Values outside the configured [min, max] ranges are run as given, not clipped, so
        the model runs exactly what the caller asked for. What is enforced is physical
        validity, whatever the entry path: every capacitance and resistance must be
        strictly positive (A is built from 1/(R*C)), and the cooling, gain and solar loads
        must not be negative (a negative load would reverse its direction).

        Parameters
        ----------
        values : dict
            {name: value} covering RC_PARAM_KEYS, in machine space. See
            scaled_params_to_tensors().

        Raises
        ------
        ValueError
            If any parameter maps to a physically invalid value.
        """
        params, loads = scaled_params_to_tensors(values, len(self.building.rooms))
        self._check_physically_valid(params, loads)
        self.params = nn.Parameter(params, requires_grad=False)
        self.loads = nn.Parameter(loads, requires_grad=False)

    def _check_physically_valid(self, params, loads):
        theta = self.transform(params) if self.transform else params
        physical_params = self.scaling.physical_param_scaling(theta).flatten()
        physical_loads = self.scaling.physical_loads_scaling(self.transform(loads) if self.transform else loads)

        problems = [
            f"{key}={physical_params[i].item():g} (must be > 0)"
            for i, key in enumerate(PARAM_KEYS)
            if not physical_params[i].item() > 0
        ]
        problems += [
            f"{key}={physical_loads[i].tolist()} (must be >= 0)"
            for i, key in enumerate(LOAD_KEYS)
            if (physical_loads[i] < 0).any()
        ]
        if problems:
            raise ValueError("Physically invalid RC parameters: " + ", ".join(problems))

    def get_parameters(self):
        """Inverse of set_parameters(): the current parameters as a machine-space dict.

        Loads are returned per room, so the result always round-trips through
        set_parameters() even when it was originally given a single value for all rooms.
        """
        values = {key: self.params[i].item() for i, key in enumerate(PARAM_KEYS)}
        for i, key in enumerate(LOAD_KEYS):
            values[key] = self.loads[i].detach().cpu().numpy().copy()
        return values

    def forward(self, t_eval, action=0):
        """
        Integrate the model forward in time with a CONSTANT action, returning the state at
        every point in t_eval.

        Because the action is held constant over the call and A/B are constant for a given
        parameter set, this is a linear time-invariant system with a known input trajectory,
        so it is stepped with an exact first-order-hold discretisation (see _foh_discretize)
        rather than a numerical ODE solver. The discretisation is cached per timestep, so the
        per-call cost is a handful of small matrix-vector products.

        See _forward_odeint() for the reference (torchdiffeq rk4) implementation this must
        agree with; test_forward_matches_odeint pins them together.

        t_eval - times the function should return a solution for, in absolute (unix epoch)
            seconds. Must be sorted ascending.
        action - 0 or 1, held constant across the whole call.
        """
        t_eval = self._prepare_forward(t_eval, action)

        u = self._input_trajectory(t_eval)
        x0 = self.iv.reshape(-1).to(torch.float64).numpy()
        t_np = t_eval.to(torch.float64).numpy()

        states = _integrate_states(
            self.A.detach().to(torch.float64).numpy(),
            self.B.detach().to(torch.float64).numpy(),
            t_np,
            u,
            x0,
            disc_cache=self._disc_cache,
        )

        self.iv = None  # Causes error if iv is not reset before next forward pass.

        # (n_steps, n_states, 1) to match what the odeint path returned - the environment squeezes it.
        return torch.tensor(states, dtype=torch.float32).unsqueeze(-1)

    def _prepare_forward(self, t_eval, action):
        """Shared bookkeeping for both forward implementations. Returns a flat t_eval."""
        self.action = action
        self.record_action = []  # Keeps track of action at time during integration.

        # check if t_eval is formatted correctly:
        if t_eval.dim() != 1:
            t_eval = t_eval.flatten()

        # t0 stores the starting epoch time; used by plotting and by _forward_odeint.
        self.t0 = t_eval[0]

        if self.cooling_policy:
            # The action is constant across the call, so start and end fully describe it.
            # (Only used for plotting - see tools/plotting.py.)
            self.record_action.append([0.0, self.action])
            self.record_action.append([(t_eval[-1] - self.t0).item(), self.action])

        # Format iv.
        self.iv = self.iv.reshape((2 + len(self.building.rooms), 1)).to(torch.float32)

        return t_eval

    def _input_trajectory(self, t_eval):
        """
        Build the input vector u = [Tout, Q_rm1, ... Q_rmn] at every point in t_eval.

        The action is held constant across the call, so the cooling and gain part of Q is
        constant; the solar part follows GHI(t). Crucially Tout and GHI are each fetched for
        the whole window in ONE batched interpolation call rather than once per solver
        sub-step, which is where most of the old cost went.

        Returns a (len(t_eval), n_rooms + 1) float64 array, matching Building.input_vector()'s
        ordering.
        """
        u = np.empty((len(t_eval), len(self.building.rooms) + 1), dtype=np.float64)
        u[:, 0] = _sample(self.Tout_continuous, t_eval)
        u[:, 1:] = self._heat_input_watts(t_eval, self.action)
        return u

    def _heat_input_watts(self, t, action):
        """
        Heat input into each room, in W, at every time in t: floor area * (gain - cool *
        action + solar * GHI(t)). Returns a (len(t), n_rooms) float64 array.
        """
        area = np.array([room.area for room in self.building.rooms], dtype=np.float64)
        constant = (self.gain_load - self.cool_load * action).detach().to(torch.float64).numpy()
        q_area = np.broadcast_to(constant, (len(t), len(area))).copy()  # W/m2

        solar = self.solar_load
        if solar is not None:
            solar = solar.detach().to(torch.float64).numpy()
            if np.any(solar != 0):
                q_area += self.ghi(t)[:, None] * solar[None, :]

        return q_area * area[None, :]

    def ghi(self, t):
        """Global horizontal irradiance (W/m2) at absolute times t, as a float64 array.

        Zeros if the model was built without solar data.
        """
        ghi_continuous = getattr(self, "ghi_continuous", None)  # absent on models pickled before solar
        if ghi_continuous is None:
            return np.zeros(len(t), dtype=np.float64)
        return _sample(ghi_continuous, t)

    def _forward_odeint(self, t_eval, action=0):
        """
        Reference implementation of forward() using torchdiffeq's fixed-step rk4.

        Kept ONLY as the thing forward() is validated against (see
        test_forward_matches_odeint) - it is not used in training, where it was far too slow
        for a population-based search: it re-interpolated Tout at every solver sub-step.
        """
        t_eval = self._prepare_forward(t_eval, action)
        t_eval = t_eval - self.t0

        integrate = odeint(self.f_ode, self.iv, t_eval, method="rk4")  # https://github.com/rtqichen/torchdiffeq

        self.iv = None  # Causes error if iv is not reset before next forward pass.

        return integrate  # first two columns are external envelope nodes. i.e not rooms

    def f_ode(self, t, x):
        """
        Provides the function:
        dy/dx = Ax + Bu

        Only used by _forward_odeint().
        """
        # Get energy input at timestep:
        t_abs = t.item() + self.t0
        t_query = torch.as_tensor(t_abs, dtype=torch.float64).reshape(1)
        Q_watts = torch.tensor(self._heat_input_watts(t_query, self.action)[0], dtype=torch.float32)

        Tout = self.Tout_continuous(t_abs)

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

        self.cool_load = loads[LOAD_KEYS.index("cool"), :]
        self.gain_load = loads[LOAD_KEYS.index("gain"), :]
        self.solar_load = loads[LOAD_KEYS.index("solar"), :]

    def initialise_parameters(self):
        """Initialise params and loads with random values in 0-1 machine space.

        A PBT run overwrites these immediately via set_parameters(); this only matters for a
        model built without an explicit parameter set.
        """
        params = torch.rand(self.building.n_params, dtype=torch.float32)
        loads = torch.rand((len(LOAD_KEYS), len(self.building.rooms)), dtype=torch.float32)

        # enables spread of initial parameters. Otherwise, sigmoid(rand) tends towards 0.5.
        if self.transform == torch.sigmoid:
            params = torch.logit(params)  # inverse sigmoid
            loads = torch.logit(loads)

        # make theta torch parameters. No gradients: parameters are searched, not descended.
        self.params = nn.Parameter(params, requires_grad=False)

        # initialise the room cooling and gain loads
        self.loads = nn.Parameter(loads, requires_grad=False)

    def slowest_time_constant(self):
        """
        The slowest time constant of the current parameter set, in seconds.

        The eigenvalues of A are the system's modes; the one closest to the imaginary axis
        decays slowest, and tau = -1/Re(lambda) is how long it takes to settle. The physical
        parameter ranges legally permit R*C products giving time constants of hundreds of
        days, and a building that slow simply cannot be identified from a few weeks of data -
        a PBT trial holding such a draw is wasted compute. See pbt.RCPolicyTrainable's
        max_time_constant_days option, which uses this to sideline those trials cheaply.

        Returns inf if the system has a non-decaying (zero or positive real part) mode.

        Requires A to have been built - call setup() (or _build_matrices()) first.
        """
        if self.A is None:
            raise RuntimeError("Call setup() before slowest_time_constant() - A has not been built.")

        eigvals = np.linalg.eigvals(self.A.detach().to(torch.float64).numpy())
        real_parts = eigvals.real

        # A stable thermal system has strictly negative real parts. Anything else never
        # settles, which is as useless as being arbitrarily slow.
        if np.any(real_parts >= 0):
            return float("inf")

        return float(1.0 / np.min(np.abs(real_parts)))

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
        Go from the stored machine-space parameters (0-1 by default) to the true physical values.
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


def _sample(continuous, t):
    """Evaluate a weather callable (Interp1D or plain function) at t as a flat float64 array.

    A constant callable may return a scalar; it is broadcast to len(t).
    """
    values = torch.as_tensor(continuous(t)).flatten().to(torch.float64)
    if values.numel() == 1:
        values = values.repeat(len(t))
    return values.detach().numpy()


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


def _step_recurrence(Ad, Bd0, Bd1, u, x0):
    """
    Apply x_{k+1} = Ad @ x_k + Bd0 @ u_k + Bd1 @ u_{k+1} for a CONSTANT (Ad, Bd0, Bd1).

    A plain loop, because the windows this runs over are short (one environment step is
    typically tens of samples) and each iteration is one small matrix-vector product. The
    filter-based decomposition used by _integrate_latent_vectorized only pays off over long
    trajectories with few input channels; here there is one input per room, so it would cost
    more than it saves.

    u: (n_steps, m) input samples. x0: (n,) initial state.
    Returns: (n_steps, n) state trajectory including x0 at index 0.
    """
    n_steps = len(u)
    X = np.empty((n_steps, len(x0)))
    X[0] = x0
    x = x0
    for k in range(n_steps - 1):
        x = Ad @ x + Bd0 @ u[k] + Bd1 @ u[k + 1]
        X[k + 1] = x
    return X


def _integrate_states(A, B, t, u, x0, disc_cache=None):
    """
    Propagate the exact FOH discretization of dx/dt = A x + B u(t) across every point in `t`.

    `disc_cache` is an optional dict owned by the caller (RCModel keeps one, cleared whenever
    A/B change) so the matrix exponential is computed once per timestep rather than once per
    environment step. It is keyed by dt.

    Note on exactness: the FOH assumption is that u is linear BETWEEN consecutive points of
    `t`. The cooling and gain part of Q is constant, so that only constrains Tout and GHI (the
    solar part of Q is proportional to GHI) - it holds whenever the data grid is at least as
    fine as the weather grids they are interpolated from (30s indoor data vs hourly
    temperature and 15-minute irradiance, say). If a weather series were finer than `t`, a
    step could span a kink in it and this would become an approximation rather than an
    identity.
    """
    if len(t) < 2:
        return np.repeat(np.asarray(x0, dtype=np.float64)[None, :], len(t), axis=0)

    dt_arr = np.diff(t)
    uniform = np.allclose(dt_arr, dt_arr[0], rtol=_UNIFORM_DT_TOL, atol=_UNIFORM_DT_TOL)

    if uniform:
        dt = float(dt_arr[0])
        cached = disc_cache.get(dt) if disc_cache is not None else None
        if cached is None:
            cached = _foh_discretize(A, B, dt)
            if disc_cache is not None:
                disc_cache[dt] = cached
        Ad, Bd0, Bd1 = cached
        return _step_recurrence(Ad, Bd0, Bd1, u, x0)

    # dt varies between samples: discretize every step with its own dt in one batched call
    # rather than silently reusing a single-dt discretization that would be wrong elsewhere.
    Ad_b, Bd0_b, Bd1_b = _foh_discretize_batch(A, B, dt_arr)
    n_steps = len(u)
    X = np.empty((n_steps, len(x0)))
    X[0] = x0
    x = np.asarray(x0, dtype=np.float64)
    for k in range(n_steps - 1):
        x = Ad_b[k] @ x + Bd0_b[k] @ u[k] + Bd1_b[k] @ u[k + 1]
        X[k + 1] = x
    return X


def _integrate_latent_vectorized(Ad, Bd0, Bd1, W, x0):
    """
    Fully vectorized propagation of x_{k+1} = Ad @ x_k + Bd0 @ w_k + Bd1 @ w_{k+1}
    for a CONSTANT (Ad, Bd0, Bd1), i.e. uniform dt. Decomposes the response into a
    per-input-channel IIR filter (the exact digital-filter realization of this LTI
    system, run via scipy) plus the closed-form free response of x0 via
    eigendecomposition of Ad - avoids a Python loop over time steps entirely.

    Used by get_iv_array(), where the trajectory is the whole dataset (long) and there are
    only two input channels - the regime where this beats a plain loop.

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
