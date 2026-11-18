import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import lu_factor, lu_solve


def cheb_points(n):
    x = np.arange(n+1)*np.pi/n
    return np.cos(x)

def chebyshev_diff_matrix(N: int):
    if N == 0:
        D = np.array([[0.0]])
        x = np.array([1.0])
        return D, x

    x = cheb_points(N)

    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c = c * (-1) ** np.arange(N + 1)

    X = np.tile(x.reshape(-1, 1), (1, N + 1))
    dX = X - X.T

    D = (c.reshape(-1, 1) / c.reshape(1, -1)) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))

    return D, x



def cahn_hilliard(N, eps=0.01, nsteps=1000, dt=1e-4):
    D, nodes = chebyshev_diff_matrix(N)
    
    # Scale D by Jacobian
    D = 4 * D

    D2 = D@D

    mapping = lambda x: (1 + x) / 2
    x = mapping(nodes)

    CN = np.cos(np.pi*x)
    WN = np.zeros(len(CN))
    I = np.eye(N+1)
    A1 = -dt * D2
    A2 = -2 * I/eps + eps * D2

    # Impose B.C.s
    I[0, :] = D[0, :]
    I[-1, :] = D[-1, :]
    A1[0, :] = 0
    A1[-1, :] = 0
    A2[0, :] = 0
    A2[-1, :] = 0
    

    LHS = np.bmat([[I, A1], [A2, I]])
    lu, piv = lu_factor(LHS)

    for i in range(int(nsteps)):
        CN[0] = 0
        CN[-1] = 0

        RHS = np.hstack([CN, (CN**3 - 3 * CN) / eps])
        sol = lu_solve((lu, piv), RHS)
        CN = sol[:N+1]
        WN = sol[N+1:]
    return x, CN