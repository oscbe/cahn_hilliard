import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsp
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
from scipy.integrate import trapezoid

class CH2D:
    def __init__(self, N, eps=0.01, dt=None):
        self.N = N
        self.dx = 1/(N-1)
        self.eps = eps
        self.phi = lambda x: ((1 - x**2)**2 )/ 4

        self.t = 0
        if dt is None:
            self.dt = self.dx**4
        else:
            self.dt = dt

        self.mesh = np.linspace(0, 1, N)
        X, Y = np.meshgrid(self.mesh, self.mesh)

        rng = np.random.default_rng(0)
        self.Un = rng.uniform(-1, 1, X.flatten().shape)


        self.Dx, self.Dy = self.construct_derivs()
        self.D2 = self.construct_laplace_2d()

        self.I = spsp.eye(N*N, N*N)
        self.compute_lu()

    def construct_derivs(self):
        deriv_1d = spsp.diags([-1, 1], [-1, 1], (self.N, self.N), dtype='float64').tolil()
        # deriv_1d[0, 0] = 1
        # deriv_1d[-1, -1] = 1
        # deriv_1d[0, :] *= 2
        # deriv_1d[-1, :] *= 2

        deriv_1d[0, 0] = -3
        deriv_1d[0, 1] = 4
        deriv_1d[0, 2] = -1
        
        deriv_1d[-1, -1] = -3
        deriv_1d[-1, -2] = 4
        deriv_1d[-1, -3] = -1

        deriv_1d /= 2 * self.dx
        I = spsp.eye(self.N)

        return spsp.kron(I, deriv_1d), spsp.kron(deriv_1d, I)

    def construct_laplace_2d(self):
        laplace_1d = spsp.diags([1, -2, 1], [-1, 0, 1], (self.N, self.N), dtype='float64').tolil()
        laplace_1d[0, 1] *= 2
        laplace_1d[-1, -2] *= 2

        laplace_1d /= self.dx**2

        return spsp.kronsum(laplace_1d, laplace_1d, format='csr')
    
    def compute_lu(self):
        A = spsp.bmat([
            [self.I, -self.dt * self.D2],
            [self.eps * self.D2 - 2*self.I/self.eps, self.I]
        ])

        # self.lu, self.piv = lu_factor(A.todense())
        self.lu = splu(A)

    def step(self):
        Cn = self.Un[:(self.N * self.N)]
        rhs2 = (Cn**3 - 3*Cn) / self.eps
        # self.Un = lu_solve((self.lu, self.piv), np.hstack([Cn, rhs2]))
        self.Un = self.lu.solve(np.hstack([Cn, rhs2]))
        self.t += self.dt

    def plot_state(self):
        c = self.Un[:(self.N * self.N)]
        c = c.reshape([self.N, self.N])

        plt.cla()
        plt.imshow(c, extent=[0, 1, 0, 1], origin='lower', vmin=-1.0, vmax=1.0, cmap='RdBu_r')
        plt.show()

    def compute_M(self):
        c = self.Un[:(self.N * self.N)].reshape((self.N, self.N))

        return trapezoid(trapezoid(c, self.mesh), self.mesh)
    
    def compute_J(self):
        c = self.Un[:self.N**2]

        cx = self.Dx@c
        cy = self.Dy@c

        F = (self.phi(c)/self.eps + self.eps * (cx**2 + cy**2) / 2).reshape((self.N, self.N))

        return trapezoid(trapezoid(F, self.mesh), self.mesh)



if __name__ == "__main__":
    import time

    N = 128
    dt = 1e-4
    T = 0.6
    n_steps = int(T / dt)
    print("Total steps:", n_steps)
    sample_every = 10

    solver = CH2D(N, dt=dt)
    
    sol_pics_on = list((n_steps/2**np.array([0, 2, 4, 6, 8, 10], dtype=int)[::-1]).astype(int) - 1)

    Ms = []
    Js = []
    ts = []
    sols = []
    sols_ts = []
    m0 = solver.compute_M()
    t = time.time()
    for i in range(0, n_steps):
        solver.step()
        if i % sample_every == 0:
            Ms.append(solver.compute_M() - m0)
            Js.append(solver.compute_J())
            ts.append(solver.t)
        if i in sol_pics_on:
            print(f"Completed {i} of {n_steps} iterations")
            sols.append(solver.Un)
            sols_ts.append(solver.t)
    print("Seconds elapsed:", time.time() - t)

    solver.plot_state()

    # for sol in sols:
    #     c = sol[:N**2].reshape((N, N))
    #     plt.cla()
    #     plt.imshow(c, extent=[0, 1, 0, 1], origin='lower', vmin=-1.0, vmax=1.0, cmap='RdBu_r')
    #     plt.show()

    fig, axs = plt.subplots(2, 3, constrained_layout=True)
    axs = axs.flatten()

    for i, sol in enumerate(sols):
        ax = axs[i]
        c = sol[:N**2].reshape((N, N))
        im = ax.imshow(c, extent=[0, 1, 0, 1], origin='lower', vmin=-1.0, vmax=1.0, cmap='RdBu_r')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"t = {sols_ts[i]:.2}", fontweight='bold')

    cbar = fig.colorbar(im, ax=axs, shrink=0.8, location='right')
    plt.savefig("figs/two_dim_sol")
    plt.show()
    plt.close(fig)

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
    fig.suptitle(f"Quantities of Computed Solution", fontweight='bold')
    plt.tight_layout()
    fig.savefig("figs/two_dim_mass")
    plt.show()