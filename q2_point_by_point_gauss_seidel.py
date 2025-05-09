# Import package
import numpy as np
import time as time

# Define a function to get the Numerical solution using gauss seidel iterative method
def point_by_point_gauss_seidel(ini):
    
    # Problem parameters
    L, H = 1.0, 2.0
    dx, dy = 0.05, 0.05
    
    # Nodes
    nx = int(L/dx + 1)
    ny = int(H/dy + 1)

    # Matrix for phi and coefficients 
    
    phi = np.ones((nx, ny)) * ini

    Ae = np.ones((nx, ny))
    Aw = np.ones((nx, ny))
    An = np.ones((nx, ny))
    As = np.ones((nx, ny))
    Ap = np.ones((nx, ny)) * (Ae + Aw + An + As)
    
    # Set boundary face coefficients to 0
    Aw[0, :] = 0    # West
    As[:, 0] = 0    # South
    Ae[-1, :] = 0   # East
    An[:, -1] = 0   # North

    # Boundary conditions
    phi[:, 0]  = 100.0  # bottom
    phi[:, -1] = 0.0    # top
    phi[0, :]  = 0.0    # left
    phi[-1, :] = 0.0    # right


    # Define the convergence criterion
    tolerance = 1e-4
    max_iter = 10000
    iteration = 0
    error = 1

    # Start timing the solver
    start_time = time.time()

    # While loop with 2 conditions, so it will stop when either of these become false either error < tolerance or max iterations is reached
    while error > tolerance and iteration < max_iter:
        error = 0
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                phi_old = phi[i, j]
                phi[i, j] = (Ae[i, j] * phi[i+1, j] + Aw[i, j] * phi[i-1, j] +
                             An[i, j] * phi[i, j+1] + As[i, j] * phi[i, j-1]) / Ap[i, j]
                error = max(error, abs(phi[i, j] - phi_old))
        iteration += 1

    print("Gauss-Seidel iterations:", iteration)
    
    # End timing thesolver
    end_time = time.time()
    print(f"Gauss seidel solver time: {end_time - start_time:.6f} seconds.")
    return phi
