import numpy as np
import matplotlib.pyplot as plt

# Polar plots for angular distributions
theta = np.linspace(0, 2*np.pi, 360)
sigma_electric = 1e-3 * (np.sin(theta/2)**2 + 0.5*np.cos(theta/2)**2)
sigma_magnetic = 5e-4 * (np.cos(theta/2)**2 + 0.3*np.sin(theta/2)**2)

plt.figure(figsize=(10, 8))
ax = plt.subplot(111, polar=True)

# Plot both scattering patterns on the same graph
ax.plot(theta, sigma_electric, color='blue', linewidth=2, label='Electric Dipole')
ax.plot(theta, sigma_magnetic, color='red', linewidth=2, label='Magnetic Dipole')

# Customize the plot
ax.set_title('Electric and Magnetic Dipole Scattering\nComparison', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.tight_layout()
plt.savefig('combined_dipole_scattering_polar.png', dpi=300, bbox_inches='tight')
plt.show()
