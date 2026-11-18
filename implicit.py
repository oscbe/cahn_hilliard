import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable, Optional, Literal
from scipy import sparse as spsp
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
from scipy.integrate import trapezoid


class SolverMixin:
    def newton_vec(self,
        xk: Union[float, int],
        F: Callable,
        J: Callable,
        damping: Union[float, int] = 1,
        tol: float = 1e-14,
        max_iter: int = 100,
        return_iter_count: bool = False
        ) -> float:
        r"""Use the Newton-Raphson algorithm to find a root of a given function from a starting point.

        Args
        ----
            xl: float or int
                point at which to start the algorithm
            f: callable
                function to find a roots of
            df: callable
                derivative of arg `f`
            damping: float or int
                scaling applied to the search distance
            tol: float
                desired tolerance of the root (how close to zero it is)
            max_iter: int
                maximum number of iterations to search for a root
            return_iter_count: bool
                indicates if the iteration count should be returned
        """
        k = 0
        while (abs(F(xk)).sum() > tol) and (k < max_iter):
            xk = xk + damping * spsp.linalg.spsolve(J(xk), -F(xk))
            k += 1
        
        if return_iter_count:
            return xk, k
        return xk

class CahnHilliard1D(SolverMixin):
    def __init__(self,
                N: int,
                init_cond: Callable,
                phi: Callable,
                phi_p: Callable,
                phi_pp: Callable,
                boundary_type: Literal['Neumann', 'Dirichlet'] = 'Neumann',
                eps: float = 0.01,
                dt: Optional[float] = None
                ):
        self.N = N
        self.dx = 1/(N-1)
        self.init_cond = init_cond
        self.phi = phi
        self.phi_p = phi_p
        self.phi_pp = phi_pp
        self.boundary_type = boundary_type
        self.eps = eps
        self.t = 0

        if dt == None:
            self.dt = self.dx**4
        else:
            self.dt = dt
        self.mu = self.dt / (self.dx**2)

        self.mesh = np.linspace(0, 1, N)

        C0 = init_cond(self.mesh)
        W0 = np.zeros(C0.shape)

        self.Un = np.concat([C0, W0])
        self.A = self.construct_A()
        self.mat_base = self._make_mat_base()

        self._make_J_mat()
        self.J = self.compute_J()

        self.M = self.compute_M()

    def construct_A(self):
        "Construct tridiagonal matrix representing discrete second derivative"
        A = spsp.diags([np.ones(self.N-1), -2*np.ones(self.N), np.ones(self.N-1)], [-1, 0, 1])

        if self.boundary_type == 'Neumann':
            A += spsp.csc_matrix(([1, 1], [(0, self.N-1), (1, self.N-2)]), [self.N, self.N])
        elif self.boundary_type == 'Dirichlet':
            raise NotImplementedError("Dirichlet B.C.s not yet implemented")
        return A

    def _make_mat_base(self):
        I = spsp.eye(self.N)
        return spsp.block_array([[I, -self.A*self.mu], [self.eps * self.A/(self.dx**2), I]])
    
    def step(self):
        F = lambda x: self.mat_base@x - np.concat([self.Un[:self.N], self.phi_p(x[:self.N])/self.eps])
        J = lambda y: self.mat_base - spsp.diags(self.phi_pp(y[:self.N]), -self.N)/self.eps
        self.Un = self.newton_vec(self.Un, F, J, tol=1e-13, max_iter=30)
        self.t += self.dt

    def _make_J_mat(self):

        dneg1 = -np.ones(self.N - 1)
        dneg1[-1] = -2

        d0 = np.zeros(self.N)
        d0[0] = -2
        d0[-1] = 2

        dpos1 = np.ones(self.N - 1)
        dpos1[0] = 2

        self._J_mat = spsp.diags([dneg1, d0, dpos1], [-1, 0, 1]) / (2 * self.dx)
    
    def compute_J(self):
        c = self.Un[:self.N]
        term_1 = 1/self.eps * self.phi(c)
        term_2 = self.eps * (self._J_mat@c)**2 / 2

        return trapezoid(term_1 + term_2, self.mesh)
        return (term_1 + term_2).sum()
    
    def compute_M(self):
        c = self.Un[:self.N]
        return trapezoid(c, self.mesh)
        return self.dx * (c[1:] + c[:-1]).sum()
    

class SchemeA(CahnHilliard1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phi_p = lambda x, y: -y+x**3
        self.phi_pp = lambda x, y: 3*x**2
    
    def step(self):
        F = lambda x: self.mat_base@x - np.concat([self.Un[:self.N], self.phi_p(x[:self.N], self.Un[:self.N])/self.eps])
        J = lambda y: self.mat_base - spsp.diags(self.phi_pp(y[:self.N], self.Un[:self.N]), -self.N)/self.eps
        self.Un = self.newton_vec(self.Un, F, J, tol=1e-13, max_iter=20)

class SchemeB(CahnHilliard1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_lu()

    def step(self):
        Cn = self.Un[:self.N]
        rhs2 = (Cn**3 - 3*Cn) / self.eps
        self.Un = lu_solve((self.lu, self.piv), np.hstack([Cn, rhs2]))
        # self.Un = self.lu.solve(np.hstack([Cn, rhs2]))
        self.t += self.dt

    def compute_lu(self):
        coeffs = 2 * np.ones(self.N) / self.eps
        A = self.mat_base - spsp.diags(coeffs, -self.N)
        self.lu, self.piv = lu_factor(A.todense()) # no need for sparsity as LU should be efficient
        # self.lu = splu(A)

# class SchemeB(CahnHilliard1D):
#     def step(self):
#         Cn = self.Un[:self.N]
#         coeffs = 2 * np.ones(len(Cn)) / self.eps
#         rhs2 = (Cn**3 - 3*Cn)/self.eps
#         A = self.mat_base - spsp.diags(coeffs, -self.N)
#         self.Un = spsp.linalg.spsolve(A, np.hstack([Cn, rhs2]))
#         self.t += self.dt

class SchemeC(CahnHilliard1D):
    def step(self):
        Cn = self.Un[:self.N]
        coeffs = (3*(Cn**2) - 1) / self.eps
        rhs2 = -2*Cn**3/self.eps
        A = self.mat_base - spsp.diags(coeffs, -self.N)
        self.Un = spsp.linalg.spsolve(A, np.hstack([Cn, rhs2]))
        self.t += self.dt

class SchemeD(CahnHilliard1D):
    def step(self):
        Cn = self.Un[:self.N]
        coeffs = Cn**2 / self.eps
        rhs2 = -Cn/self.eps
        A = self.mat_base - spsp.diags(coeffs, -self.N)
        self.Un = spsp.linalg.spsolve(A, np.hstack([Cn, rhs2]))
        self.t += self.dt


if __name__ == '__main__':
    import time
    N = 128

    phi = lambda x: ((1-x**2)**2)/4
    phi_p = lambda x: x**3 - x
    phi_pp = lambda x: 3*x**2 - 1
    base_dt = (1/(N-1))**4

    all_Cs = []
    all_Ws = []
    all_Ms = []
    all_Js = []
    elapseds = []
    schemes = {
        # "Fully Implicit": CahnHilliard1D,
        # "Scheme A": SchemeA,
        "Scheme B": SchemeB,
        # "Scheme C": SchemeC,
        # "Scheme D": SchemeD
        }
    for scheme_name, scheme in schemes.items():
        dt = 1e-4
        T = 0.2
        n_steps = int(T / dt)
        print(n_steps)
        sample_every = 1

        # if scheme_name == "Scheme B":
        #     dt = dt / 10
        #     n_steps = 10 * n_steps
        #     sample_every = 10 * sample_every

        print("Running with", scheme_name)
        solver = scheme(N, lambda x: np.cos(np.pi * x), phi, phi_p, phi_pp, dt=dt)
        vals = []
        Js = []
        Ms = []
        ts = []

        t = time.time()
        m0 = solver.compute_M()
        for i in range(n_steps):
            solver.step()
            if i % sample_every == 0:
                vals.append(solver.Un[:N])
                ts.append(solver.t)
                Js.append(solver.compute_J())
                Ms.append(solver.compute_M() - m0)

        all_Cs.append(solver.Un[:N])
        all_Ws.append(solver.Un[N:])
        all_Ms.append(Ms)
        all_Js.append(Js)

        elapsed = time.time() - t
        print(elapsed)
        print(solver.t)
        elapseds.append(elapsed)


        # plt.imshow(np.array(vals), cmap='hot', interpolation='nearest', origin='lower', extent=[0, 1, ts[0], ts[-1]], aspect='auto')
        # plt.clim(-1, 1)
        # plt.colorbar()
        # plt.ylabel('Time')
        # plt.xlabel('x')
        # plt.title('Evolution of Computed Solution to\nCahn-Hilliard Equation, Implicit Scheme')
        # plt.show()

    # for i in range(len(all_Cs)):
    fig, axs = plt.subplots(1, 2)
    fig.set_figwidth(10)
    for i in range(0, len(all_Cs)):
        axs[0].plot(solver.mesh, all_Cs[i], label=list(schemes.keys())[i])
        axs[1].plot(solver.mesh, all_Ws[i], label=list(schemes.keys())[i])

    axs[0].set(xlabel='x', ylabel='C')
    axs[0].set_ylim([-1.05, 1.05])
    axs[0].grid(True)
    axs[0].legend()
    axs[1].set(xlabel='x', ylabel='W')
    axs[1].grid(True)
    axs[1].legend()
    fig.suptitle(f"Computed Solutions\n$t={solver.t:.2}$", fontweight='bold')
    plt.tight_layout()
    # fig.savefig("figs/implicit_sol_metastable")
    plt.show()
    plt.close(fig)


    fig, axs = plt.subplots(1, 2)
    fig.set_figwidth(10)
    for i in range(0, len(all_Cs)):
        axs[0].plot(ts, all_Ms[i], label=list(schemes.keys())[i])
        axs[1].plot(ts, all_Js[i], label=list(schemes.keys())[i])
    axs[0].set(xlabel='Time', ylabel='Change in Mass')
    axs[0].grid(True)
    axs[0].set_xticks(axs[0].get_xticks()[1::2])
    axs[0].legend()
    axs[1].set(xlabel='Time', ylabel='J')
    axs[1].grid(True)
    axs[1].set_xticks(axs[1].get_xticks()[1::2])
    axs[1].legend()
    fig.suptitle(f"Quantities of Computed Solutions", fontweight='bold')
    plt.tight_layout()
    # fig.savefig("figs/implicit_mass_metastable")
    plt.show()
    print(elapseds)