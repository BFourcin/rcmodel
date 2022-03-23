import torch
import numpy as np
from torchdiffeq import odeint


def latent_sim(model, t_eval, Tin_continuous):

    bl = model.building

    # Recalculate the A & B matrices. We could chop and re jig from the full matrices, but it is not super simple
    # so recalculating is less risky.
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

    t0 = t_eval[0]
    t_eval = t_eval - t0

    def f_ode(t, x):
        Tout = model.Tout_continuous(t.item() + t0)
        Tin = Tin_continuous(t.item() + t0)

        u = torch.tensor([[Tout],
                          [Tin]])

        return A @ x + B @ u.to(torch.float32)

    integrate = odeint(f_ode, model.iv[0:2], t_eval,
                       method='rk4')  # https://github.com/rtqichen/torchdiffeq

    return integrate

