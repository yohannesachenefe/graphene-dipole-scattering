# Reproduce Figure 2: Interference Heatmap

import numpy as np
import matplotlib.pyplot as plt
from graphene_scattering import *

print("Reproducing Figure 2 from the paper...")

# Parameters
energy = 0.1  # eV
soft_core_radius = 0.1  # nm

# Define parameter sweep
dipole_strengths = np.linspace(0.1, 1.0, 50)  # Total dipole strength
theta = np.linspace(0, 2*np.pi, 200)

# Initialize arrays
cross_sections = np.zeros((len(dipole_strengths), len(theta)))

# Calculate for each dipole strength
for i, p in enumerate(dipole_strengths):
    # Split strength between electric and magnetic
    d_electric = np.array([p * 0.7, 0.0])  # 70% electric
    d_magnetic = np.array([0.0, p * 0.3, 0.0])  # 30% magnetic
    
    sim = ScatteringSimulation(energy=energy)
    sim.set_parameters(
        electric_dipole=d_electric,
        magnetic_dipole=d_magnetic,
        soft_core_radius=soft_core_radius
    )
    
    cross_sections[i, :] = sim.differential_cross_section(theta)
    
    if i % 10 == 0:
        print(f"  Progress: {i/len(dipole_strengths)*100:.0f}%")

# Convert to normalized units
hbar = 6.582119569e-16  # eV·s
vF = 1e6  # m/s
conversion_factor = (hbar**2 * vF**2) * 1e18
cross_sections_norm = cross_sections * conversion_factor

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Convert theta to degrees
theta_deg = np.degrees(theta)

# Create meshgrid
THETA, PARAM = np.meshgrid(theta_deg, dipole_strengths)

# Plot
im = ax.pcolormesh(THETA, PARAM, cross_sections_norm, 
                  cmap='viridis', shading='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r'$\frac{d\sigma}{d\theta}$ ($\hbar^{-2} v_F^{-2}$)', 
               rotation=270, labelpad=20, fontsize=12)

# Labels and title
ax.set_xlabel('Scattering Angle (degrees)', fontsize=12)
ax.set_ylabel(r'Total Dipole Strength $p$ ($e\cdot$nm + $\mu_B$)', fontsize=12)
ax.set_title('Figure 2: Interference Effects in Combined Dipole Scattering', 
             fontsize=14, pad=20)

# Set x-ticks
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_xlim([0, 360])

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figure2_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print(f"\nHeatmap statistics:")
print(f"  Dipole strength range: {dipole_strengths[0]:.2f} to {dipole_strengths[-1]:.2f}")
print(f"  Cross section range: {np.min(cross_sections_norm):.2e} to {np.max(cross_sections_norm):.2e}")
print(f"  Mean cross section: {np.mean(cross_sections_norm):.2e}")