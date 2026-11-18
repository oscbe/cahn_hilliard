import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsp
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
from scipy.integrate import trapezoid



class CH2D:
    def __init__(self, img, eps=0.05, delta=10_000, dt=None):
        self.N = N = img.shape[0]
        self.dx = 1/(N-1)
        self.eps = eps
        self.phi = lambda x: ((1 - x**2)**2 )/ 4
        
        self.f = img.flatten()
        self.dmg_mask = np.logical_and(img != 1, img != -1).flatten()
        # self.no_dmg_mask = np.logical_not(self.dmg_mask).flatten()
        self.delta = delta

        self.t = 0
        if dt is None:
            self.dt = self.dx**4
        else:
            self.dt = dt

        self.mesh = np.linspace(0, 1, N)
        X, Y = np.meshgrid(self.mesh, self.mesh)

        # rng = np.random.default_rng(0)
        # self.Un = rng.uniform(-1, 1, X.flatten().shape)
        self.Un = np.hstack([self.f, np.zeros(self.f.shape)])


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
            [self.I + self.dt * self.delta * spsp.diags(self.dmg_mask.astype('float64')), -self.dt * self.D2],
            [self.eps * self.D2 - 2*self.I/self.eps, self.I]
        ])

        # self.lu, self.piv = lu_factor(A.todense())
        self.lu = splu(A.tocsc())

    def step(self):
        Cn = self.Un[:(self.N * self.N)]
        rhs2 = (Cn**3 - 3*Cn) / self.eps
        # self.Un = lu_solve((self.lu, self.piv), np.hstack([Cn, rhs2]))
        self.Un = self.lu.solve(np.hstack([Cn + self.delta * self.f * self.dmg_mask, rhs2]))
        self.t += self.dt

    def plot_state(self):
        c = self.Un[:(self.N * self.N)]
        c = c.reshape([self.N, self.N])

        plt.cla()
        plt.imshow(c, extent=[0, 1, 0, 1], origin='lower', vmin=-1.0, vmax=1.0, cmap='binary')
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
    
    def set_eps(self, eps):
        self.eps = eps
        self.compute_lu()