import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu
import time


N = 100
h = 1.0 / N
epsilon = 0.01
dt = 1e-7
T = 0.0015
n_steps = int(T / dt)
save_interval = 20


np.random.seed(33)
c_current = 0.01 * (np.random.rand((N+1)**2) - 0.5)  # initial perturbation


def get_laplacian(N, h):
    data = np.ones((3, N+1))
    data[1, :] = -2
    L1D = sparse.spdiags(data, [-1, 0, 1], N+1, N+1).tolil()
    L1D[0, 1], L1D[-1, -2] = 2, 2   # Neumann BC
    L1D = L1D * (1.0 / h**2)
    
    return L1D

D = get_laplacian(N, h)
I = sparse.eye(N+1)

# Scheme B
J21 = (epsilon * D) - (2.0 / epsilon) * I
A = sparse.bmat([
    [I, -dt * D],
    [J21,  I]
], format='csr')
lu = splu(A)

def get_trapezoidal_weights(N):
    W = np.ones(N+1)     # interior
    
    # boundaries (0.5)
    W[0]  *= 0.5
    W[-1] *= 0.5
    # vertices 0.25 automatically
    
    return W.flatten()

weights_flat = get_trapezoidal_weights(N)


mass_history = []
energy_history = []
time_points = []

# reset c_current
np.random.seed(33)
c_current = 0.01 * (np.random.rand(N+1) - 0.5) 
mesh = np.linspace(0, 1, N+1)
c_current = np.cos(np.pi * mesh)


m0 = np.sum(c_current * weights_flat) * h

print(f"Total number of steps: {n_steps}")
start_time = time.time()

for step in range(n_steps):
    rhs2 = (1.0 / epsilon) * (c_current**3 - 3.0 * c_current)
    rhs = np.concatenate([c_current, rhs2])
    
    sol = spsolve(A, rhs)
    # sol = lu.solve(rhs)
    
    c_new = sol[:N+1]
    w_new = sol[N+1:]
    
    c_current = c_new
    current_mass = np.sum(c_current * weights_flat) * h
    
    phi = (1.0 / (4.0 * epsilon)) * (c_current**2 - 1.0)**2
    bulk_energy = np.sum(phi * weights_flat) * h
    
    lap_c = D.dot(c_current)
    integrand_grad = c_current * lap_c 
    grad_energy = -0.5 * epsilon * np.sum(integrand_grad * weights_flat) * h
    total_energy = bulk_energy + grad_energy
    
    mass_history.append(current_mass - m0)
    energy_history.append(total_energy)
    time_points.append(step * dt)
        
    if (step + 1) % (n_steps // 5) == 0:
        print(f"We are at {(step+1)/n_steps*100:.0f}%. Energy: {total_energy:.4e}")

print(f"Time consuming: {time.time() - start_time:.2f}s")
plt.plot(c_current)
plt.show()

plt.plot(mass_history)
plt.show()

plt.plot(energy_history)
plt.show()