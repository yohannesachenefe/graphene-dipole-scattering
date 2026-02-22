import numpy as np
import matplotlib.pyplot as plt
# 2D Heatmap of scattering cross-section
angles = np.linspace(0, 2*np.pi, 180)
dipole_strengths = np.linspace(0, 5, 100)
cross_section = np.outer(np.sin(angles/2)**2 + 0.2*np.cos(angles/2)**2, 
                         dipole_strengths**2)

plt.figure(figsize=(8, 6))
extent = [dipole_strengths.min(), dipole_strengths.max(), 0, 360]
plt.imshow(cross_section.T, extent=extent, aspect='auto', 
           origin='lower', cmap='viridis')
plt.colorbar(label='Differential Cross Section (a.u.)')
plt.xlabel('Dipole Strength (arbitrary units)')
plt.ylabel('Scattering Angle (degrees)')
plt.title('2D Heatmap: Cross Section vs. Angle and Dipole Strength')
plt.savefig('heatmap_cross_section.png', dpi=300, bbox_inches='tight')
plt.show()
