# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Import functions
from q2_Analytical_sol import analytical_solution
from q2_point_by_point_gauss_seidel import point_by_point_gauss_seidel
from q2_line_by_line_tdma import line_by_line_TDMA

def main():
    # Compute solutions using different methods
    phi_gs   = point_by_point_gauss_seidel(ini=40)
    phi_tdma = line_by_line_TDMA(ini=40)
    X, Y, phi_ana = analytical_solution(Nmax=40)

    # Set domain parameters and grid points for plotting
    L, H = 1.0, 2.0
    dx, dy = 0.05, 0.05
    nx = int(L/dx + 1)
    ny = int(H/dy + 1)
    x_vals = np.linspace(0, L, nx)
    y_vals = np.linspace(0, H, ny)

    # --- Plot the contour plots ---
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    cs1 = plt.contourf(X, Y, phi_gs, 20, cmap='jet')
    plt.colorbar(cs1)
    plt.title("Gauss-Seidel")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.subplot(1,3,2)
    cs2 = plt.contourf(X, Y, phi_tdma, 20, cmap='jet')
    plt.colorbar(cs2)
    plt.title("Line-by-line TDMA")
    plt.xlabel("x (m)")

    plt.subplot(1,3,3)
    cs3 = plt.contourf(X, Y, phi_ana, 20, cmap='jet')
    plt.colorbar(cs3)
    plt.title("Analytical (Nmax=8)")
    plt.xlabel("x (m)")
    plt.tight_layout()
    plt.show()
    
    # --- Generate 1D profile plots ---

    # Axial profiles: Temperature vs. y at x = 0.25, 0.5, 0.75 m
    axial_positions = [0.25, 0.5, 0.75]
    axial_indices = [int(pos/dx) for pos in axial_positions]

    for x_idx, x_val in zip(axial_indices, axial_positions):
        plt.figure()
        plt.plot(y_vals, phi_gs[x_idx, :], '--', label='Gauss-Seidel')
        plt.plot(y_vals, phi_tdma[x_idx, :], '-.', label='TDMA')
        plt.plot(y_vals, phi_ana[x_idx, :], '-', label='Analytical')
        plt.xlabel("y (m)")
        plt.ylabel("Temperature (°C)")
        plt.title(f"Temperature vs. y at x = {x_val} m")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Radial profiles: Temperature vs. x at y = 0.5, 1.0, 1.5 m
    radial_positions = [0.5, 1.0, 1.5]
    radial_indices = [int(pos/dy) for pos in radial_positions]

    for y_idx, y_val in zip(radial_indices, radial_positions):
        plt.figure()
        plt.plot(x_vals, phi_gs[:, y_idx], '--', label='Gauss-Seidel')
        plt.plot(x_vals, phi_tdma[:, y_idx], '-.', label='TDMA')
        plt.plot(x_vals, phi_ana[:, y_idx], '-', label='Analytical')
        plt.xlabel("x (m)")
        plt.ylabel("Temperature (°C)")
        plt.title(f"Temperature vs. x at y = {y_val} m")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
