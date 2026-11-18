import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

class ExplicitCHSolver:
    def __init__(self, N, eps=0.01):
        self.mesh = np.linspace(0, 1, N)
        self.eps = eps
        self.nx = N
        self.dx = self.mesh[1] - self.mesh[0]
        self.C0 = np.cos(np.pi * self.mesh)
        self.phi_p = lambda c: c**3 - c
        self.phi = lambda c: ((1 - c**2)**2) / 4
        self.W0 = None

        self.Dxx = np.diag(-2*np.ones(self.nx)) \
            + np.diag(np.ones(self.nx - 1), -1) \
                + np.diag(np.ones(self.nx - 1), 1)
        
        self.Dxx[0, 1] *= 2
        self.Dxx[-1, -2] *= 2

        self.dt = self.dx**4

        self.reset()

        self.Dx = (np.diag(np.ones(self.nx - 1), 1) - np.diag(np.ones(self.nx - 1), -1)) * 0.5
        self.Dx[0, 0] = -1
        self.Dx[0, 1] = 1
        self.Dx[-1, -1] = 1
        self.Dx[-1, -2] = -1


    def reset(self):
        self.CN = self.C0
        self.WN = self.phi_p(self.CN) - self.Dxx@self.CN
        self.t = 0

    def step(self):
        self.CN = self.CN + self.dt * (self.Dxx@self.WN) / (self.dx**2)
        self.WN = self.phi_p(self.CN) / self.eps - self.eps * (self.Dxx@self.CN)/(self.dx**2)

        self.t += self.dt

    def plot(self):
        plt.plot(self.mesh, self.CN)
        plt.show()

    def compute_M(self):
        return trapezoid(self.CN, self.mesh)
    
    def compute_J(self):
        grad_c = self.Dx@self.CN
        F = self.phi(self.CN) / self.eps + self.eps * grad_c * grad_c / 2
        return trapezoid(F, self.mesh)


    def run(self, T, max_steps=10000, plot_every=False):

        i = 0
        while ((self.t < T) and (i < max_steps)):
            i += 1
            self.step()
            if (plot_every and (i % plot_every == 0)):
                self.plot()

if __name__ == "__main__":
    # import time
    # solver = ExplicitCHSolver(128)
    # M0 = solver.compute_M()
    # J0 = solver.compute_J()
    # Ms = [M0]
    # Js = [J0]
    # ts = [0]
    # t = time.time()
    # for i in range(4000000):
    #     solver.step()
    #     if i % 10000 == 0:
    #         Ms.append(solver.compute_M() - M0)
    #         Js.append(solver.compute_J())
    #         ts.append(solver.t)
    # elapsed = time.time() - t
    # print(elapsed)

    # fig, axs = plt.subplots(1, 2)
    # fig.set_figwidth(10)
    # axs[0].plot(solver.mesh, solver.CN)
    # axs[0].set(xlabel='x', ylabel='C')
    # axs[0].set_ylim([-1.05, 1.05])
    # axs[0].grid(True)
    # axs[1].plot(solver.mesh, solver.WN)
    # axs[1].set(xlabel='x', ylabel='W')
    # axs[1].grid(True)
    # fig.suptitle(f"Computed Solution with Explicit Scheme\n$t={solver.t:.2}$", fontweight='bold')
    # plt.tight_layout()
    # # fig.savefig("figs/explicit_sol")
    # plt.show()
    # plt.close(fig)


    # fig, axs = plt.subplots(1, 2)
    # fig.set_figwidth(10)
    # axs[0].plot(ts, Ms)
    # axs[0].set(xlabel='Time', ylabel='Change in Mass')
    # axs[0].grid(True)
    # axs[1].plot(ts, Js)
    # axs[1].set(xlabel='Time', ylabel='J')
    # axs[1].grid(True)
    # fig.suptitle(f"Quantities of Computed Solution with Explicit Scheme", fontweight='bold')
    # plt.tight_layout()
    # fig.savefig("figs/explicit_mass")
    # # plt.show()

    # from scipy.interpolate import CubicSpline
    # Ns = [16, 32, 64]#, 128]
    # meshes = []
    # sols = []
    # for N in Ns:
    #     print("Running with N = ", N)
    #     solver = ExplicitCHSolver(N)
    #     solver.run(0.1, max_steps=int(1e+8))
    #     sols.append(solver.CN)
    #     meshes.append(solver.mesh)
    # print("Solutions computed.")
    
    # exact = CubicSpline(meshes[-1], sols[-1])
    # fine_grid = np.linspace(0, 1, 1000)
    # errs = []
    # for i, sol in enumerate(sols[:-1]):
    #     interpolated = CubicSpline(meshes[i], sol)
    #     err = abs(interpolated(fine_grid) - exact(fine_grid))
    #     errs.append(err.max())

    # plt.loglog(Ns[:-1], errs, label='Error')
    # hs = np.array([(1 / N**2) for N in Ns[:-1]])
    # factor = 1.01 * errs[0] / hs[0]
    # hs *= factor
    # plt.plot(Ns[:-1], hs, linestyle='--', label='$O((\\Delta x)^2)$')
    # plt.gca().set(xlabel='Number of Mesh Points', ylabel='Error ($L^\\infty$ Norm)')
    # plt.title("Error for Different Mesh Sizes", fontweight='bold')
    # plt.legend()
    # plt.show()



    N = 128
    n_steps = 60000
    sample_every = 400

    solver = ExplicitCHSolver(N)
    vals = []
    J_vals = []
    M_vals = []
    ts = []
    for i in range(n_steps):
        solver.step()
        if i % sample_every == 0:
            vals.append(solver.CN)
            ts.append(solver.t)
            J_vals.append(solver.compute_J())
            M_vals.append(solver.compute_M())

    # plt.plot(solver.mesh, list(zip(*vals)))
    # plt.ylabel('c')
    # plt.xlabel('x')
    # plt.show()

    plt.imshow(np.array(vals), cmap='hot', interpolation='nearest', origin='lower', extent=[0, 1, ts[0], ts[-1]], aspect='auto')
    # plt.xticks([0, 25, 50, 75, 99], [0, 0.25, 0.5, 0.75, 1])
    plt.clim(-1, 1)
    plt.colorbar()
    plt.ylabel('Time')
    plt.xlabel('x')
    plt.title('Evolution of Computed Solution to\nCahn-Hilliard Equation, Explicit Scheme', fontweight='bold')
    plt.tight_layout()
    # plt.savefig('figs/explicit_heatmap')
    plt.show()