# Import packages
import numpy as np
import time as time

def line_by_line_TDMA(ini):
    # Problem parameters
    L, H = 1.0, 2.0
    dx, dy = 0.05, 0.05

    # Grid nodes
    nx = int(L/dx + 1)
    ny = int(H/dy + 1)

    # Initialize phi and coefficient matrices
    phi = np.ones((nx, ny)) * ini

    Ae = np.ones((nx, ny))
    Aw = np.ones((nx, ny))
    An = np.ones((nx, ny))
    As = np.ones((nx, ny))

    # Set boundary face coefficients to zero
    Aw[0, :] = 0    # West
    Ae[-1, :] = 0   # East
    As[:, 0] = 0    # South
    An[:, -1] = 0   # North

    Ap = Ae + Aw + An + As

    # Apply Dirichlet boundary conditions
    phi[:, -1] = 0.0  # North
    phi[:, 0] = 100.0 # South
    phi[0, :] = 0.0   # West
    phi[-1, :] = 0.0  # East

    # Convergence parameters
    max_iter = 10000
    tolerance = 1e-4
    iteration = 0
    error = 1.0
    
    
    # Start timing the TDMA solver
    start_time = time.time()
    
    # Iterative solver using line-by-line TDMA
    while iteration < max_iter and error > tolerance:
        phi_old = phi.copy()
        error = 0.0

        for i in range(1, nx - 1):
            # Prepare TDMA coefficients for current line i
            a = np.zeros(ny)
            b = np.zeros(ny)
            c = np.zeros(ny)
            d = np.zeros(ny)

            for j in range(1, ny - 1):
                d[j] = Ae[i, j] * phi_old[i + 1, j] + Aw[i, j] * phi_old[i - 1, j]
                if j == 1:
                    d[j] += As[i, j] * 100  # South boundary condition
                    c[j] = 0.0
                else:
                    c[j] = As[i, j]
                if j == ny - 2:
                    d[j] += An[i, j] * 0.0  # North boundary condition
                    b[j] = 0.0
                else:
                    b[j] = An[i, j]
                a[j] = Ap[i, j]

            # TDMA sweep for the current line i
            P = np.zeros(ny)
            Q = np.zeros(ny)
            for j in range(1, ny - 1):
                denom = a[j] - c[j] * P[j - 1]
                P[j] = b[j] / denom
                Q[j] = (d[j] + c[j] * Q[j - 1]) / denom

            for j in range(ny - 2, 0, -1):
                phi[i, j] = P[j] * phi[i, j + 1] + Q[j]

            current_error = np.max(np.abs(phi[i, 1:-1] - phi_old[i, 1:-1]))
            if current_error > error:
                error = current_error

        iteration += 1
        if error < tolerance:
            print(f"Line-by-line TDMA converged in {iteration} iterations.")
            break

    # End timing the TDMA solver
    end_time = time.time()
    print(f"TDMA solver time: {end_time - start_time:.6f} seconds.")
    
    return phi
