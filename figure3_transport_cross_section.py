# Reproduce Figure 3: Transport Cross Section

import numpy as np
import matplotlib.pyplot as plt
from graphene_scattering import *

print("Reproducing Figure 3 from the paper...")

# Parameters
energy = 0.1  # eV
soft_core_radius = 0.1  # nm

# Dipole strength range
dipole_moments = np.logspace(-2, 0, 30)  # 0.01 to 1.0

# Initialize arrays
sigma_tr_electric = np.zeros_like(dipole_moments)
sigma_tr_magnetic = np.zeros_like(dipole_moments)

# Calculate for each dipole moment
for i, d in enumerate(dipole_moments):
    # Electric dipole (along x)
    sim_e = ScatteringSimulation(energy=energy)
    sim_e.set_parameters(electric_dipole=[d, 0.0])
    sigma_tr_electric[i] = sim_e.transport_cross_section()
    
    # Magnetic dipole (in-plane along y)
    sim_m = ScatteringSimulation(energy=energy)
    sim_m.set_parameters(magnetic_dipole=[0.0, d, 0.0],
                        soft_core_radius=soft_core_radius)
    sigma_tr_magnetic[i] = sim_m.transport_cross_section()
    
    if i % 5 == 0:
        print(f"  Progress: {i/len(dipole_moments)*100:.0f}%")

# Convert to normalized units
hbar = 6.582119569e-16  # eV·s
vF = 1e6  # m/s
conversion_factor = (hbar**2 * vF**2) * 1e18

sigma_tr_e_norm = sigma_tr_electric * conversion_factor
sigma_tr_m_norm = sigma_tr_magnetic * conversion_factor

# Create Figure 3
fig, ax = plt.subplots(figsize=(8, 6))

# Plot data points
ax.loglog(dipole_moments, sigma_tr_e_norm, 'bo', markersize=6, 
          label='Electric Dipole', alpha=0.7)
ax.loglog(dipole_moments, sigma_tr_m_norm, 'rs', markersize=6, 
          label='Magnetic Dipole', alpha=0.7)

# Plot analytical scaling laws
# For electric dipole: σ_tr ∝ d²
# For magnetic dipole: σ_tr ∝ m² (with soft-core modification)
d_fit = np.logspace(-2, 0, 100)
sigma_e_fit = 0.1 * d_fit**2  # Arbitrary prefactor to match scale
sigma_m_fit = 0.05 * d_fit**2  # Smaller prefactor for magnetic

ax.loglog(d_fit, sigma_e_fit, 'b-', alpha=0.5, linewidth=1.5, 
          label=r'$d^2$ scaling (electric)')
ax.loglog(d_fit, sigma_m_fit, 'r-', alpha=0.5, linewidth=1.5,
          label=r'$m^2$ scaling (magnetic)')

# Labels and title
ax.set_xlabel(r'Dipole Moment Magnitude ($e\cdot$nm or $\mu_B$)', fontsize=12)
ax.set_ylabel(r'$\sigma_{\mathrm{tr}}$ ($\hbar^{-2} v_F^{-2}$)', fontsize=12)
ax.set_title('Figure 3: Transport Cross Section vs Dipole Moment', 
             fontsize=14, pad=20)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('figure3_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()

# Print comparison
print(f"\nTransport cross section comparison:")
print(f"  At dipole moment = 0.5:")
idx = np.argmin(np.abs(dipole_moments - 0.5))
print(f"    Electric: {sigma_tr_e_norm[idx]:.2e}")
print(f"    Magnetic: {sigma_tr_m_norm[idx]:.2e}")
print(f"    Ratio (E/M): {sigma_tr_e_norm[idx]/sigma_tr_m_norm[idx]:.2f}")