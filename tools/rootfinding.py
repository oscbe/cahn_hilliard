import numpy as np
from typing import Union, Callable

def newton_vec(
        xk: Union[float, int],
        F: Callable,
        J: Callable,
        damping: Union[float, int] = 1,
        tol: float = 1e-14,
        max_iter: int = 100000,
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
        xk = xk + damping * np.linalg.solve(J(xk), -F(xk))
        k += 1
    
    if return_iter_count:
        return xk, k
    return xk