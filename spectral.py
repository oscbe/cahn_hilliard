import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import BarycentricInterpolator, PchipInterpolator, CubicSpline
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

def differentiation_matrix(nodes):
    """
    Return the spectral differentiation matrix for arbitrary 1D nodes.

    Parameters
    ----------
    nodes : array-like
        Distinct node locations x_0, x_1, ..., x_{n-1}

    Returns
    -------
    D : (n, n) ndarray
        Differentiation matrix such that f'(x_i) ≈ sum_j D[i, j] f(x_j)
    """
    x = np.asarray(nodes, dtype=float)
    n = x.size

    if n < 2:
        raise ValueError("Need at least two distinct nodes.")
    if len(np.unique(x)) != n:
        raise ValueError("All nodes must be distinct.")

    # Barycentric weights
    w = np.ones(n)
    for j in range(n):
        diff = x[j] - np.delete(x, j)
        w[j] = 1.0 / np.prod(diff)

    # Differentiation matrix
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = w[j] / (w[i] * (x[i] - x[j]))
        D[i, i] = -np.sum(D[i, :])

    return D

def double_chebyshev(N):
    base = cheb_points(N)
    m1 = (base - 1) / 2
    m2 = (base + 1) / 2
    mesh = np.hstack([m2, m1[1:]])
    return differentiation_matrix(mesh), mesh


def cahn_hilliard(N, eps=0.01, nsteps=1000, dt=1e-4):
    D, nodes = chebyshev_diff_matrix(N)
    D, nodes = double_chebyshev(N)
    # Scale D by Jacobian
    N = 2 * N
    D = 2 * D

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

class CHSpectral:
    def __init__(self, N, eps=0.01, dt=0.01):
        self.N = N
        D, nodes = chebyshev_diff_matrix(N)
        self.D = 2 * D
        self.D2 = self.D@self.D
        self.x = (1 + nodes) / 2
        self.eps = eps
        self.dt = dt
        self.phi = lambda x: ((1 - x**2)**2)/4
        self.fine_mesh = np.linspace(0, 1, 200)
        self.t = 0


        self.CN = np.cos(np.pi*self.x)
        self.WN = np.zeros(len(self.CN))
        I = np.eye(N+1)
        A1 = -dt * self.D2
        A2 = -2 * I/eps + eps * self.D2

        # Impose B.C.s
        I[0, :] = D[0, :]
        I[-1, :] = D[-1, :]
        A1[0, :] = 0
        A1[-1, :] = 0
        A2[0, :] = 0
        A2[-1, :] = 0

        LHS = np.bmat([[I, A1], [A2, I]])
        self.lu, self.piv = lu_factor(LHS)

    def step(self):
        self.CN[0] = 0
        self.CN[-1] = 0

        RHS = np.hstack([self.CN, (self.CN**3 - 3 * self.CN) / self.eps])
        sol = lu_solve((self.lu, self.piv), RHS)
        self.CN = sol[:self.N+1]
        self.WN = sol[self.N+1:]
        self.t += self.dt

    def interp(self):
        self.c_interpolant = BarycentricInterpolator(self.x, self.CN)
        self.w_interpolant = BarycentricInterpolator(self.x, self.WN)

    def compute_J(self):
        self.interp()
        term_1 = self.phi(self.c_interpolant(self.fine_mesh)) / self.eps
        grad_c = self.c_interpolant.derivative(self.fine_mesh)
        term_2 = grad_c**2 * self.eps / 2

        return trapezoid(term_1 + term_2, self.fine_mesh)
    
    def compute_M(self):
        self.interp()
        return trapezoid(self.c_interpolant(self.fine_mesh), self.fine_mesh)





def find_boundary(N):
    A = np.zeros(N + 1, dtype=bool)
    A[0] = True
    A[-1] = True
    edges = np.kron(A, np.ones(N+1, dtype=bool)) | np.kron(np.ones(N+1, dtype=bool), A)
    corners = np.kron(A, A)
    return edges, corners

def cahn_hilliard_2d(N, eps=0.1, nsteps=1000, dt=1e-4):
    D, nodes = chebyshev_diff_matrix(N)
    
    # Scale D by Jacobian
    D = 2 * D

    D2 = D@D

    mapping = lambda x: (1 + x) / 2
    x = mapping(nodes)
    X, Y = np.meshgrid(x, x)

    # CN = np.cos(np.pi*X) * np.cos(np.pi*Y)
    CN = np.random.uniform(-1, 1, X.shape)
    CN = CN.flatten()
    WN = np.zeros(CN.shape)
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
    
    IK = np.eye(N+1)

    LHS = np.bmat([
        [np.kron(I, IK) + np.kron(IK, I), np.kron(A1, IK) + np.kron(IK, A1)],
        [np.kron(A2, IK) + np.kron(IK, A2), np.kron(I, IK) + np.kron(IK, I)]
        ])
    lu, piv = lu_factor(LHS)

    print(np.kron(IK, IK))

    edges, corners = find_boundary(N)
    edges = np.hstack([edges, edges])

    for i in range(int(nsteps)):
        # CN[edges] = 0
        # CN[edges] = 0
        # Don't need to do it with corners as well, as they are covered by edges

        RHS = np.hstack([CN, (CN**3 - 3 * CN) / eps])
        RHS[edges] = 0
        sol = lu_solve((lu, piv), RHS)
        CN = sol[:((N+1) * (N+1))]
        WN = sol[((N+1) * (N+1)):]

    return X.flatten(), Y.flatten(), CN, WN



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x_vals = np.linspace(0, 1, 1000)
    # Ns = [32, 64, 128]
    # Ns = [8, 16, 18]
    # cs = ['tab:blue', 'tab:orange', 'black']
    # alphas = [0.7, 0.7, 1]

    # for i, N in enumerate(Ns):
    #     sol = BarycentricInterpolator(*cahn_hilliard(N, eps=0.01, nsteps=3, dt=0.01))
    #     # sol = CubicSpline(*cahn_hilliard(N, eps=0.01, nsteps=3, dt=0.01))
    #     plt.plot(x_vals, sol(x_vals), label=f"N = {2*N + 1}", c=cs[i], alpha=alphas[i])
    #     # plt.plot(x_vals, sol(x_vals))
    # plt.legend()
    # plt.grid()
    # t = 1000 * 0.01
    # plt.gca().set(xlabel='$x$', ylabel='$c$')
    # plt.title(f"Runge Phenomenon with Spectral Methods,\n$T=0.03$", fontweight='bold')
    # # plt.savefig("figs/spectral_runge")
    # plt.show()


    solver = CHSpectral(128, dt=0.1)
    Ms = []
    Js = []
    ts = []
    # for i in range(10):
    #     solver.step()
    M0 = solver.compute_M()
    solver.compute_J()
    for i in range(100):
        solver.step()
        if i % 1 == 0:
            Ms.append(solver.compute_M() - M0)
            Js.append(solver.compute_J())
            ts.append(solver.t)

    plt.plot(solver.x, solver.CN)
    plt.show()

    fig, axs = plt.subplots(1, 2)
    fig.set_figwidth(10)
    axs[0].plot(ts, Ms)
    axs[1].plot(ts, Js)
    axs[0].set(xlabel='Time', ylabel='Change in Mass')
    axs[0].grid(True)
    axs[0].set_xticks(axs[0].get_xticks()[1::2])
    axs[1].set(xlabel='Time', ylabel='J')
    axs[1].grid(True)
    axs[1].set_xticks(axs[1].get_xticks()[1::2])
    fig.suptitle(f"Quantities of Computed Solutions", fontweight='bold')
    plt.tight_layout()
    # fig.savefig("figs/spectral_mass")
    plt.show()