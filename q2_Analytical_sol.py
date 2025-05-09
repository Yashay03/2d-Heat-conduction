# Import Package
import numpy as np

# Define a function to calculate analytical solution with number of terms as a parameter
def analytical_solution(Nmax):
    
    # Problem parameters
    L, H = 1.0, 2.0
    dx, dy = 0.05, 0.05
    
    # Calculation of nodes
    nx = int(L/dx + 1)
    ny = int(H/dy + 1)
    
    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    T_ana = np.zeros_like(X)

    # for loop over odd k = 2n-1
    for n in range(1, Nmax+1):
        k = 2*n - 1
        term = (4/(k*np.pi) * np.sin(k*np.pi*X/L) * np.sinh(k*np.pi*(H - Y)/L) / np.sinh(k*np.pi*H/L))
        
        # Adds the terms to the analytical solution
        T_ana += term

    T_ana *= 100.0
    return X, Y, T_ana 
