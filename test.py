import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import lu_factor, lu_solve


def cheb_points(n):
    return np.cos(np.pi * np.arange(n + 1) / n)


def chebyshev_diff_matrix(N: int):
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])

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


def cahn_hilliard_2d_fixed(N, eps=0.1, nsteps=1000, dt=1e-4, seed=0):
    """
    Mixed 2D Cahn-Hilliard solver on [0,1]x[0,1] with no-flux BCs:
        d_n c = 0
        d_n w = 0

    Unknown ordering:
        u = [c, w]
    Vectorization:
        Fortran order, so vec(U) = U.ravel(order='F')
    """
    D, nodes = chebyshev_diff_matrix(N)

    # Map [-1, 1] -> [0, 1]
    D = 2.0 * D
    D2 = D @ D
    x = 0.5 * (1 + nodes)

    nx = N + 1
    ny = N + 1
    npts = nx * ny

    X, Y = np.meshgrid(x, x, indexing="ij")

    # 2D operators in Fortran ordering
    Ix = np.eye(nx)
    Iy = np.eye(ny)

    D2x = np.kron(Iy, D2)
    D2y = np.kron(D2, Ix)
    Lap = D2x + D2y

    Dx = np.kron(Iy, D)
    Dy = np.kron(D, Ix)

    I2 = np.eye(npts)

    # Initial condition: random uniform in [-1, 1]
    rng = np.random.default_rng(seed)
    CN = rng.uniform(-1, 1, (nx, ny))
    CN = CN.ravel(order="F")
    WN = np.zeros_like(CN)

    # Mixed-form block system
    A1 = -dt * Lap
    A2 = -(2.0 / eps) * I2 + eps * Lap

    LHS = np.block([
        [I2, A1],
        [A2, I2]
    ]).astype(float)

    # Right-hand-side vector
    RHS = np.zeros(2 * npts)

    # Boundary rows: use x-derivative on left/right edges, y-derivative on top/bottom edges.
    # Corners are handled by the first condition that matches.
    for j in range(ny):
        for i in range(nx):
            p = i + nx * j  # Fortran-order index

            if i == 0 or i == nx - 1:
                row = Dx[p, :]
            elif j == 0 or j == ny - 1:
                row = Dy[p, :]
            else:
                continue

            # Enforce d_n c = 0
            LHS[p, :] = 0.0
            LHS[p, :npts] = row
            RHS[p] = 0.0

            # Enforce d_n w = 0
            LHS[npts + p, :] = 0.0
            LHS[npts + p, npts:] = row
            RHS[npts + p] = 0.0

    lu, piv = lu_factor(LHS)

    for _ in range(int(nsteps)):
        # Mixed-form nonlinear RHS for the chemical potential equation
        RHS[:npts] = CN
        RHS[npts:] = (CN**3 - 3.0 * CN) / eps

        # Boundary RHS remains zero because BC rows were already inserted into LHS
        sol = lu_solve((lu, piv), RHS)

        CN = sol[:npts]
        WN = sol[npts:]
        yield CN.reshape((nx, ny), order="F")

    # return X, Y, CN.reshape((nx, ny), order="F"), WN.reshape((nx, ny), order="F")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 40
    # X, Y, C, W = cahn_hilliard_2d_fixed(N, eps=0.1, nsteps=500, dt=1e-4)

    # plt.figure()
    # plt.imshow(C, origin="lower", extent=[0, 1, 0, 1])
    # plt.colorbar()
    # plt.title("Cahn-Hilliard concentration")
    # plt.show()

    sols = [i for i in cahn_hilliard_2d_fixed(N, nsteps=100, dt=0.001)]
    for i in range(0, len(sols), 10):
        plt.figure()
        plt.imshow(sols[i], origin="lower", extent=[0, 1, 0, 1])
        plt.colorbar()
        plt.title("Cahn-Hilliard concentration")
        plt.show()