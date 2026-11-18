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

def barycentric_interpolant(nodes, vals):
    def f(x):
        if isinstance(x, np.ndarray):
            return np.array([f(i) for i in x])
        
        if x in nodes:
             return vals[nodes == x].item()
        num = np.array([(-1)**k * vals[k]/(x - node) for k, node in enumerate(nodes)])
        denom = np.array([(-1)**k/(x - node) for k, node in enumerate(nodes)])
        num[[0, -1]] *= 0.5
        denom[[0, -1]] *= 0.5
        return num.sum()/denom.sum()
    return f

def poisson_1d(f, N):
    D, nodes = chebyshev_diff_matrix(N)
    D2 = D@D
    D2 = D2[1: -1, 1:-1]
    sol = np.linalg.solve(D2, f(nodes[1:-1]))
    result = np.zeros(N+1)
    result[1:-1] = sol
    return nodes, result


def test_problem(N):
    g = lambda x: np.exp(- x**2)*(2 * x**2 - 1) **2 - np.exp(-1)
    D, nodes = chebyshev_diff_matrix(N)

    X = np.diag(nodes)
    LHS = (X@X) @ (D@D) + X@D + np.eye(N+1)
    sol = np.linalg.solve(LHS[1:-1, 1:-1], g(nodes[1:-1]))
    return nodes, np.concatenate([[0], sol, [0]])

def test_problem_2(N):
    D, nodes = chebyshev_diff_matrix(N)
    g = lambda x: (np.pi **2 + 1) * np.sin(np.pi * x)
    mapping = lambda x: (x + 1) / 2
    mapped_g = lambda x: g(mapping(x))

    D = 2 * D
    D2 = D@D
    LHS = -D2 + np.eye(N+1)
    sol = np.linalg.solve(LHS[1:-1, 1:-1], mapped_g(nodes[1:-1]))
    return mapping(nodes), np.concatenate([[0], sol, [0]])


def cahn_hilliard(N, eps=0.1, nsteps=1000, dt=1e-4):
    D, nodes = chebyshev_diff_matrix(N)
    D2 = D@D

    dx = abs(nodes[:-1] - nodes[1:]).min()

    CN = np.cos(np.pi*nodes)
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
    return nodes, CN


def neumann_test(N):
    D, nodes = chebyshev_diff_matrix(N)
    A = -(D@D) + np.eye(N+1)
    A[0, :] = D[0, :]
    A[-1, :] = D[-1, :]

    b = (np.pi**2 + 1) * np.cos(np.pi * nodes)
    b[0] = 0
    b[-1] = 0

    sol = np.linalg.solve(A, b)

    return nodes, sol


def report_model_problem(N):
    D, nodes = chebyshev_diff_matrix(N)
    g = lambda x: (np.pi**2 + 1) * (np.sin(np.pi * x) - np.pi * np.cos(np.pi * x)) * x - 2*np.pi*np.cos(np.pi * x) - 2*np.pi*np.pi* np.sin(np.pi * x)
    mapping = lambda x: (-x + 1) / 2
    mapped_g = lambda x: g(mapping(x))

    D = -2 * D
    D2 = D@D
    LHS = -D2 + np.eye(N+1)
    LHS[-1, :] = D[-1, :]
    e1 = np.zeros(N+1)
    e1[0] = 1
    LHS[0, :] = e1
    RHS = mapped_g(nodes)
    RHS[0] = 0
    RHS[-1] = 0
    sol = np.linalg.solve(LHS, RHS)
    return mapping(nodes), sol


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x_vals = np.linspace(-1, 1, 1000)

    f = lambda x: np.exp(4 * x)

    errs = []
    Ns = np.arange(1, 32, 1)
    for N in Ns:
        sol = BarycentricInterpolator(*poisson_1d(f, N))
        actual = lambda x: ( np.exp(4*x) - np.sinh(4)*x - np.cosh(4) ) / 16
        errs.append(abs(sol(x_vals) - actual(x_vals)).max())
    
    plt.semilogy(Ns, errs)
    plt.show()

    exact = lambda x: np.exp(-x**2) - np.exp(-1)
    errs = []
    for N in Ns:
        sol = BarycentricInterpolator(*test_problem(N))
        errs.append(abs(sol(x_vals) - exact(x_vals)).max())

    plt.semilogy(Ns, errs)
    plt.show()

    sol = BarycentricInterpolator(*cahn_hilliard(256, eps=0.1, nsteps=100, dt=0.01))
    plt.plot(x_vals, sol(x_vals))
    plt.show()


    exact = lambda x: np.cos(np.pi * x)
    sol = BarycentricInterpolator(*neumann_test(12))
    plt.plot(x_vals, sol(x_vals))
    plt.plot(x_vals, exact(x_vals))
    plt.show()

    x_vals_2 = np.linspace(0, 1, 1000)
    exact = lambda x: np.sin(np.pi * x)
    sol = BarycentricInterpolator(*test_problem_2(12))
    plt.plot(x_vals_2, sol(x_vals_2))
    plt.plot(x_vals_2, exact(x_vals_2), linestyle='-')
    plt.show()


    Ns = np.arange(2, 100, 2)
    errs = []
    exact = lambda x: (np.sin(np.pi * x) - np.pi * np.cos(np.pi * x)) * x
    for N in Ns:
        sol = BarycentricInterpolator(*report_model_problem(N))
        errs.append(abs(sol(x_vals_2) - exact(x_vals_2)).max())
    plt.scatter(Ns, errs, s=10)
    plt.plot(Ns[:8], 5000*np.exp(-2*Ns[:8]))
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.xlabel("N")
    plt.ylabel("Error ($L^\\infty$ Norm)")
    plt.title("Convergence of Spectral Methods", fontweight='bold')
    plt.show()
    # plt.plot(x_vals_2, sol(x_vals_2))
    # plt.plot(x_vals_2, exact(x_vals_2), linestyle='-')
    # plt.show()