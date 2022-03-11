import torch
import numpy as np
from torch import nn
from torchdiffeq import odeint



class LatentSim(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.t0 = 0

        self.A = None  # initialise parameters
        self.B = None
        self.Tin_continuous = None

    def forward(self, t_eval, Tin_continuous):
        self.Tin_continuous = Tin_continuous
        bl = self.model.building

        # Recalculate the A & B matrices. We could chop and re jig from the full matrices, but it is more work than just
        # one slice so recalculating is less risky.
        A = torch.zeros([2, 2])
        A[0, 0] = bl.surf_area * (-1 / (bl.Re[0] * bl.Ce[0]) - 1 / (bl.Re[1] * bl.Ce[0]))
        A[0, 1] = bl.surf_area / (bl.Re[1] * bl.Ce[0])
        A[1, 0] = bl.surf_area / (bl.Re[1] * bl.Ce[1])
        A[1, 1] = bl.surf_area * (-1 / (bl.Re[1] * bl.Ce[1]) - 1 / (bl.Re[2] * bl.Ce[1]))

        B = torch.zeros([2, 2])
        B[0, 0] = bl.surf_area / (bl.Re[0] * bl.Ce[0])
        B[1, 1] = bl.surf_area / (bl.Re[2] * bl.Ce[1])

        self.A = A
        self.B = B

        # check if t_eval is formatted correctly:
        if t_eval.dim() > 1:
            t_eval = t_eval.squeeze(0)

        self.t0 = t_eval[0]
        t_eval = t_eval - self.t0

        integrate = odeint(self.f_ode, self.model.iv[0:2], t_eval, method='rk4')  # https://github.com/rtqichen/torchdiffeq

        return integrate

    def f_ode(self, t, x):

        Tout = self.model.Tout_continuous(t.item() + self.t0)
        Tin = self.Tin_continuous(t.item() + self.t0)

        u = torch.tensor([[Tout],
                          [Tin]])

        return self.A @ x + self.B @ u.to(torch.float32)

