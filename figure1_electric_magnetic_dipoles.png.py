# Reproduce Figure 1: Electric and Magnetic Dipole Scattering

import numpy as np
import matplotlib.pyplot as plt
from graphene_scattering import *

print("Reproducing Figure 1 from the paper...")

# Parameters from the paper
energy = 0.1  # eV
k = 0.1  # nm^{-1} (converted from paper)
electric_dipole = np.array([0.5, 0.0])  # e·nm
magnetic_dipole = np.array([0.0, 0.3, 0.0])  # μ_B, in-plane
soft_core_radius = 0.1  # nm

# Create simulations
sim_electric = ScatteringSimulation(energy=energy)
sim_electric.set_parameters(electric_dipole=electric_dipole)

sim_magnetic = ScatteringSimulation(energy=energy)
sim_magnetic.set_parameters(magnetic_dipole=magnetic_dipole,
                           soft_core_radius=soft_core_radius)

# Calculate cross sections
theta = np.linspace(0, 2*np.pi, 400)
cross_section_e = sim_electric.differential_cross_section(theta)
cross_section_m = sim_magnetic.differential_cross_section(theta)

# Convert to units of ħ^{-2} v_F^{-2} as in paper
# Conversion factor from nm² to these units
hbar = 6.582119569e-16  # eV·s
vF = 1e6  # m/s
conversion_factor = (hbar**2 * vF**2) * 1e18  # Convert m² to nm²

cross_section_e_norm = cross_section_e * conversion_factor
cross_section_m_norm = cross_section_m * conversion_factor

# Create Figure 1
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, 
                               figsize=(12, 6))

# Panel (a): Electric dipole
ax1.plot(theta, cross_section_e_norm, 'b-', linewidth=2)
ax1.set_theta_zero_location('E')
ax1.set_theta_direction(1)
ax1.set_rlabel_position(22.5)
ax1.grid(True, alpha=0.3)
ax1.set_title('(a) Electric Dipole Scattering', pad=20, fontsize=12)

# Panel (b): Magnetic dipole
ax2.plot(theta, cross_section_m_norm, 'r-', linewidth=2)
ax2.set_theta_zero_location('E')
ax2.set_theta_direction(1)
ax2.set_rlabel_position(22.5)
ax2.grid(True, alpha=0.3)
ax2.set_title('(b) Magnetic Dipole Scattering', pad=20, fontsize=12)

plt.suptitle('Figure 1: Differential Cross Sections in Graphene', 
             fontsize=14, y=1.05)

# Save figure
plt.savefig('figure1_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()

# Print parameters
print(f"\nParameters used:")
print(f"  Energy: {energy} eV")
print(f"  Electric dipole: {electric_dipole} e·nm")
print(f"  Magnetic dipole: {magnetic_dipole} μ_B")
print(f"  Soft core radius: {soft_core_radius} nm")
print(f"\nCross sections in units of ħ^{-2} v_F^{-2}:")
print(f"  Electric max: {np.max(cross_section_e_norm):.4e}")
print(f"  Magnetic max: {np.max(cross_section_m_norm):.4e}")
print(f"  Electric/Magnetic ratio: {np.max(cross_section_e_norm)/np.max(cross_section_m_norm):.2f}")