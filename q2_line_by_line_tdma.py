# Import package
import numpy as np
import time as time

# Define a function to get numerical solution using line_by_line tdma method
def line_by_line_TDMA(ini):
    
    # Problem Parameters
    L, H = 1.0, 2.0
    dx, dy = 0.05, 0.05
    
    # Nodes
    nx = int(L/dx + 1)
    ny = int(H/dy + 1)

    # Initial Matrix for phi and coefficients 
    phi = np.ones((nx, ny)) * ini

    Ae = np.ones((nx, ny))
    Aw = np.ones((nx, ny))
    An = np.ones((nx, ny))
    As = np.ones((nx, ny))

    # Set boundary face coefficients to 0
    Aw[0, :] = 0    # West
    As[:, 0] = 0    # South
    Ae[-1, :] = 0   # East
    An[:, -1] = 0   # North

    Ap = Ae + Aw + An + As

    # Apply boundary conditions
    phi[:, -1] = 100  # North
    phi[0, :] = 50   # West


    # Define the convergence criterion
    max_iter = 10000
    tolerance = 1e-4
    iteration = 0
    error = 1.0

    # Start timing the TDMA solver
    start_time = time.time()

    # While loop with 2 conditions, so it will stop when either of these become false either error < tolerance or max iterations is reached
    while iteration < max_iter and error > tolerance:
        phi_old = phi.copy()
        error = 0.0

        for i in range(nx):
            # Prepare TDMA coefficients for current line i
            a = np.zeros(ny)
            b = np.zeros(ny)
            c = np.zeros(ny)
            d = np.zeros(ny)

            for j in range(1, ny-1):
                d[j] = Ae[i,j] * phi_old[i+1,j] + Aw[i,j] * phi_old[i-1,j]
                if j == 1:
                    d[j] = 0  # South boundary condition
                    c[j] = 0.0
                else:
                    c[j] = As[i,j]
                if j == ny-2:
                    d[j] += An[i,j] * 100  # North boundary condition
                    b[j] = 0.0
                else:
                    b[j] = An[i,j]
                a[j] = Ap[i,j]

            # TDMA sweep for the current line i
            P = np.zeros(ny)
            Q = np.zeros(ny)
            for j in range(ny):
                denom = a[j] - c[j] * P[j-1]
                if j == 0:
                    
                    P[j] = b[j]/a[j]
                    Q[j] = d[j]/a[j]
                else:
                    P[j] = b[j] / denom
                    Q[j] = (d[j] + c[j] * Q[j-1]) / denom

            for j in range(ny-1, 0, -1):
                if j == ny-1:
                    phi[i,j] = Q[j]
                else:
                    phi[i,j] = P[j] * phi[i,j+1] + Q[j]

            current_error = np.max(np.abs(phi[i, 1:-1] - phi_old[i, 1:-1]))
            if current_error > error:
                error = current_error

        iteration += 1
        if error < tolerance:
            print("Line-by-line TDMA iterations:", iteration)
            break

    # End timing the TDMA solver
    end_time = time.time()
    print(f"TDMA solver time: {end_time - start_time:.6f} seconds.")        
    return phi
