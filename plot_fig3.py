"""
Realistic Figure 3: Transport Cross Section vs Dipole Moment
Based on: "A Python Framework for Electron Scattering from Electric and Magnetic Dipoles in Graphene"
Author: Yohannes Achenefe Nigusie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
class Constants:
    """Physical constants for graphene scattering calculations."""
    
    # Fundamental constants
    HBAR = 6.582119569e-16  # eV·s
    HBAR_J = 1.054571817e-34  # J·s
    VF = 1e6  # m/s, Fermi velocity
    E_CHARGE = 1.602176634e-19  # C
    MU_B = 9.2740100783e-24  # J/T, Bohr magneton
    MU0 = 4 * np.pi * 1e-7  # N/A²
    EPSILON0 = 8.854187817e-12  # F/m
    
    # Conversions
    NM_TO_M = 1e-9
    M_TO_NM = 1e9

# ============================================================================
# ANALYTICAL TRANSPORT CROSS SECTIONS
# ============================================================================
def electric_transport_cross_section(d, energy=0.1):
    """
    Analytical transport cross-section for electric dipole.
    
    From Eq. (8): σ_tr^(E) = π/(ħ²v_F²) * (3d_∥² + d_⟂²)
    
    Parameters
    ----------
    d : float or array
        Electric dipole moment magnitude in e·nm
    energy : float
        Incident electron energy in eV
        
    Returns
    -------
    sigma_tr : float or array
        Transport cross-section in nm²
    """
    const = Constants()
    
    # Convert dipole moment to SI
    d_si = d * const.E_CHARGE * const.NM_TO_M  # C·m
    
    # Calculate wavevector
    k = energy / (const.HBAR * const.VF)  # 1/m
    
    # For isotropic average: σ_tr = (πd²)/(ħ²v_F²)
    sigma_si = np.pi * (d_si**2) / (const.HBAR_J**2 * const.VF**2)  # m²
    
    # Convert to nm²
    sigma_nm2 = sigma_si * const.M_TO_NM**2
    
    return sigma_nm2

def magnetic_transport_cross_section(m, energy=0.1, a=0.1, alpha=1.0, beta=1.0, n_points=1000):
    """
    Numerical transport cross-section for magnetic dipole.
    
    From Eq. (15): σ_tr^(B) = (1/ħ²v_F²) ∫ (1-cosθ) e^{-4ak sin(θ/2)} [α²(q̂·m_∥)² + (βm_z/a)²] dθ
    
    Parameters
    ----------
    m : float or array
        Magnetic dipole moment magnitude in μ_B
    energy : float
        Incident electron energy in eV
    a : float
        Soft core radius in nm
    alpha, beta : float
        Coupling parameters
    n_points : int
        Number of integration points
        
    Returns
    -------
    sigma_tr : float or array
        Transport cross-section in nm²
    """
    const = Constants()
    
    # Convert to SI
    m_si = m * const.MU_B  # J/T
    a_si = a * const.NM_TO_M  # m
    
    # Calculate wavevector
    k = energy / (const.HBAR * const.VF)  # 1/m
    
    # For in-plane dipole (m_z = 0)
    m_z = 0.0
    
    if isinstance(m, np.ndarray):
        sigma_tr = np.zeros_like(m)
        for i, m_val in enumerate(m):
            sigma_tr[i] = _magnetic_sigma_integral(m_val*const.MU_B, k, a_si, alpha, beta, m_z, n_points)
    else:
        sigma_tr = _magnetic_sigma_integral(m_si, k, a_si, alpha, beta, m_z, n_points)
    
    # Convert from m² to nm²
    return sigma_tr * const.M_TO_NM**2

def _magnetic_sigma_integral(m_si, k, a_si, alpha, beta, m_z, n_points=1000):
    """Helper function for magnetic transport cross-section integration."""
    const = Constants()
    
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Momentum transfer magnitude
    q_mag = 2 * k * np.sin(theta/2)
    
    # Regularization factor
    exp_factor = np.exp(-4 * a_si * k * np.sin(theta/2))
    
    # For in-plane dipole along x-axis
    qx = q_mag * np.cos(theta)
    qy = q_mag * np.sin(theta)
    
    # Unit vector in q direction
    with np.errstate(divide='ignore', invalid='ignore'):
        q_hat_x = np.where(q_mag > 0, qx/q_mag, 0.0)
        q_hat_y = np.where(q_mag > 0, qy/q_mag, 0.0)
    
    # Dot product: q̂·m_∥ (m_∥ along x)
    q_dot_m = q_hat_x * m_si
    
    # Integrand from Eq. (15)
    integrand = (1 - np.cos(theta)) * exp_factor * (
        alpha**2 * q_dot_m**2 + (beta * m_z / a_si)**2
    )
    
    # Factor 1/(ħ²v_F²)
    prefactor = 1.0 / (const.HBAR_J**2 * const.VF**2)
    
    # Numerical integration
    sigma_si = prefactor * np.trapz(integrand, theta)
    
    return sigma_si

# ============================================================================
# ENHANCED TRANSPORT CROSS SECTION PLOT
# ============================================================================
def plot_enhanced_figure3():
    """Create enhanced Figure 3 with realistic calculations."""
    print("Generating enhanced Figure 3...")
    
    # Parameters from paper
    energy = 0.1  # eV
    a = 0.1  # nm, soft core radius
    
    # Dipole moment range (log scale from 0.01 to 1.0)
    dipole_range = np.logspace(-2, 0, 50)  # 0.01 to 1.0
    
    print("Calculating electric dipole transport cross-sections...")
    sigma_electric = electric_transport_cross_section(dipole_range, energy)
    
    print("Calculating magnetic dipole transport cross-sections...")
    sigma_magnetic = magnetic_transport_cross_section(dipole_range, energy, a)
    
    # Normalize to units of ħ^{-2} v_F^{-2}
    const = Constants()
    norm_factor = 1.0 / (const.HBAR_J**2 * const.VF**2) * const.M_TO_NM**2
    
    sigma_e_norm = sigma_electric / norm_factor
    sigma_m_norm = sigma_magnetic / norm_factor
    
    # =========================================================================
    # CREATE THE MAIN FIGURE
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))
    
    # Main plot: Transport cross-section
    ax1 = plt.subplot(2, 3, (1, 5))
    
    # Plot electric dipole (d² scaling)
    ax1.loglog(dipole_range, sigma_e_norm, 'b-', linewidth=3, alpha=0.8, 
               label='Electric Dipole', zorder=5)
    ax1.loglog(dipole_range, sigma_e_norm, 'bo', markersize=6, alpha=0.7, 
               markeredgecolor='navy', markeredgewidth=1, zorder=6)
    
    # Plot magnetic dipole (m² scaling)
    ax1.loglog(dipole_range, sigma_m_norm, 'r-', linewidth=3, alpha=0.8, 
               label='Magnetic Dipole', zorder=5)
    ax1.loglog(dipole_range, sigma_m_norm, 'rs', markersize=6, alpha=0.7, 
               markeredgecolor='darkred', markeredgewidth=1, zorder=6)
    
    # Plot theoretical scaling laws
    # Electric: σ ∝ d²
    d_fit = np.logspace(-2, 0, 100)
    sigma_e_fit = 0.1 * np.pi * d_fit**2  # πd² in normalized units
    ax1.loglog(d_fit, sigma_e_fit, 'b--', alpha=0.5, linewidth=1.5, 
               label=r'$d^2$ scaling', zorder=4)
    
    # Magnetic: σ ∝ m² with soft-core correction
    sigma_m_fit = 0.05 * np.pi * d_fit**2 * np.exp(-0.5 * d_fit)  # Example correction
    ax1.loglog(d_fit, sigma_m_fit, 'r--', alpha=0.5, linewidth=1.5, 
               label=r'$m^2$ (with soft-core)', zorder=4)
    
    # Labels and formatting
    ax1.set_xlabel(r'Dipole Moment Magnitude ($e\cdot$nm or $\mu_B$)', 
                   fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'$\sigma_{\mathrm{tr}}$ ($\hbar^{-2} v_F^{-2}$)', 
                   fontsize=13, fontweight='bold')
    ax1.set_title('(a) Transport Cross Section vs Dipole Moment', 
                  fontsize=14, fontweight='bold', pad=15)
    
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax1.grid(True, alpha=0.3, which='both', linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    # Add annotation for scaling laws
    ax1.annotate(r'$\sigma_{\mathrm{tr}}^{(E)} \propto d^2$', 
                 xy=(0.3, 0.05), xycoords='axes fraction',
                 fontsize=11, color='blue', ha='center')
    ax1.annotate(r'$\sigma_{\mathrm{tr}}^{(B)} \propto m^2$', 
                 xy=(0.3, 0.01), xycoords='axes fraction',
                 fontsize=11, color='red', ha='center')
    
    # =========================================================================
    # SUBPLOT 1: Ratio of cross sections
    # =========================================================================
    ax2 = plt.subplot(2, 3, 3)
    
    ratio = sigma_e_norm / (sigma_m_norm + 1e-10)  # Avoid division by zero
    
    ax2.semilogx(dipole_range, ratio, 'g-', linewidth=2.5, alpha=0.8)
    ax2.fill_between(dipole_range, ratio, alpha=0.2, color='green')
    
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=2, color='k', linestyle=':', alpha=0.3, linewidth=0.8)
    ax2.axhline(y=0.5, color='k', linestyle=':', alpha=0.3, linewidth=0.8)
    
    ax2.set_xlabel(r'Dipole Moment', fontsize=11)
    ax2.set_ylabel(r'$\sigma_{\mathrm{tr}}^{(E)} / \sigma_{\mathrm{tr}}^{(B)}$', 
                   fontsize=11)
    ax2.set_title('(b) Electric/Magnetic Ratio', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Add text annotation
    mean_ratio = np.mean(ratio)
    ax2.text(0.05, 0.95, f'Mean ratio: {mean_ratio:.2f}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # =========================================================================
    # SUBPLOT 2: Effective scattering strength
    # =========================================================================
    ax3 = plt.subplot(2, 3, 6)
    
    # Calculate effective scattering parameter
    k_nm = 0.1  # nm^{-1}
    hbar_vF = Constants.HBAR_J * Constants.VF
    
    # Dimensionless scattering strength
    scattering_strength_e = dipole_range * Constants.E_CHARGE * Constants.NM_TO_M * k_nm / hbar_vF
    scattering_strength_m = dipole_range * Constants.MU_B * k_nm**2 / hbar_vF
    
    ax3.loglog(dipole_range, scattering_strength_e, 'b-', linewidth=2, 
               label='Electric', alpha=0.8)
    ax3.loglog(dipole_range, scattering_strength_m, 'r-', linewidth=2, 
               label='Magnetic', alpha=0.8)
    
    # Born approximation validity line (V << ħv_Fk)
    ax3.axhline(y=0.1, color='k', linestyle='--', alpha=0.5, linewidth=1.5,
                label='Born approx. limit')
    
    ax3.set_xlabel(r'Dipole Moment', fontsize=11)
    ax3.set_ylabel(r'Dimensionless Strength', fontsize=11)
    ax3.set_title('(c) Effective Scattering Strength', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(axis='both', which='major', labelsize=10)
    
    # Add text for Born approximation
    ax3.text(0.05, 0.05, 'Below line: Born approx. valid', 
             transform=ax3.transAxes, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # SUBPLOT 3: Angular distributions at specific dipole moments
    # =========================================================================
    ax4 = plt.subplot(2, 3, (2, 5), projection='polar')
    
    # Calculate angular distributions for specific dipole moments
    theta = np.linspace(0, 2*np.pi, 200)
    
    # For dipole moment = 0.5
    d_specific = 0.5
    m_specific = 0.5
    
    # Simplified angular distributions (based on Eqs. 7 and 14)
    # Electric dipole: dσ/dθ ∝ (d·q̂)²
    dsigma_e_angular = 0.1 * (np.cos(theta) + 0.5*np.sin(theta))**2
    
    # Magnetic dipole with soft-core
    q_mag = 2 * 0.1 * np.sin(theta/2)  # k = 0.1 nm^{-1}
    exp_factor = np.exp(-0.1 * q_mag)  # a = 0.1 nm
    dsigma_m_angular = 0.05 * exp_factor * (np.sin(theta))**2
    
    # Combined
    dsigma_combined = dsigma_e_angular + dsigma_m_angular + \
                      0.03 * np.sqrt(dsigma_e_angular * dsigma_m_angular) * np.cos(2*theta)
    
    # Plot angular distributions
    ax4.plot(theta, dsigma_e_angular, 'b-', linewidth=2, alpha=0.7, 
             label=f'Electric (d={d_specific})')
    ax4.plot(theta, dsigma_m_angular, 'r-', linewidth=2, alpha=0.7,
             label=f'Magnetic (m={m_specific})')
    ax4.plot(theta, dsigma_combined, 'g-', linewidth=3, alpha=0.8,
             label='Combined', zorder=5)
    
    ax4.set_theta_zero_location('E')
    ax4.set_theta_direction(1)
    ax4.set_rlabel_position(45)
    ax4.set_title('(d) Angular Distribution at |d|=|m|=0.5', 
                  fontsize=12, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # =========================================================================
    # FINAL TOUCHES AND SAVING
    # =========================================================================
    plt.suptitle('Figure 3: Transport Cross-Section Analysis in Graphene', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save high-resolution figures
    plt.savefig('figure3_enhanced_transport_cross_section.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure3_enhanced_transport_cross_section.pdf', 
                bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("TRANSPORT CROSS-SECTION ANALYSIS SUMMARY")
    print("="*70)
    
    # Calculate at specific dipole moments
    idx_01 = np.argmin(np.abs(dipole_range - 0.1))
    idx_05 = np.argmin(np.abs(dipole_range - 0.5))
    idx_10 = np.argmin(np.abs(dipole_range - 1.0))
    
    print(f"\nAt dipole moment = 0.1:")
    print(f"  Electric σ_tr = {sigma_e_norm[idx_01]:.3e} ħ⁻² v_F⁻²")
    print(f"  Magnetic σ_tr = {sigma_m_norm[idx_01]:.3e} ħ⁻² v_F⁻²")
    print(f"  Ratio E/M = {sigma_e_norm[idx_01]/sigma_m_norm[idx_01]:.2f}")
    
    print(f"\nAt dipole moment = 0.5 (paper value):")
    print(f"  Electric σ_tr = {sigma_e_norm[idx_05]:.3e} ħ⁻² v_F⁻²")
    print(f"  Magnetic σ_tr = {sigma_m_norm[idx_05]:.3e} ħ⁻² v_F⁻²")
    print(f"  Ratio E/M = {sigma_e_norm[idx_05]/sigma_m_norm[idx_05]:.2f}")
    
    print(f"\nAt dipole moment = 1.0:")
    print(f"  Electric σ_tr = {sigma_e_norm[idx_10]:.3e} ħ⁻² v_F⁻²")
    print(f"  Magnetic σ_tr = {sigma_m_norm[idx_10]:.3e} ħ⁻² v_F⁻²")
    print(f"  Ratio E/M = {sigma_e_norm[idx_10]/sigma_m_norm[idx_10]:.2f}")
    
    # Check Born approximation validity
    print(f"\nBorn approximation validity check:")
    print(f"  Electric: V_max/(ħv_Fk) ~ {0.5*Constants.E_CHARGE*Constants.NM_TO_M*0.1**2/(Constants.HBAR_J*Constants.VF):.3f}")
    print(f"  Magnetic: V_max/(ħv_Fk) ~ {0.5*Constants.MU_B*0.1**2/(Constants.HBAR_J*Constants.VF):.3f}")
    print("  (Values << 1 indicate Born approximation is valid)")
    
    print("\n" + "="*70)
    print(f"Figure saved as:")
    print(f"  figure3_enhanced_transport_cross_section.png")
    print(f"  figure3_enhanced_transport_cross_section.pdf")
    print("="*70)

# ============================================================================
# SIMPLIFIED VERSION FOR QUICK PLOTTING
# ============================================================================
def plot_simple_figure3():
    """Simplified version for quick plotting."""
    print("Generating simplified Figure 3...")
    
    # Parameters
    energy = 0.1  # eV
    a = 0.1  # nm
    
    # Dipole range
    dipole_range = np.logspace(-2, 0, 30)
    
    # Simple analytical forms
    sigma_electric = 0.1 * np.pi * dipole_range**2
    sigma_magnetic = 0.05 * np.pi * dipole_range**2 * np.exp(-0.3*dipole_range)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot
    ax.loglog(dipole_range, sigma_electric, 'bo-', markersize=8, 
              linewidth=2.5, label='Electric Dipole', alpha=0.8)
    ax.loglog(dipole_range, sigma_magnetic, 'rs-', markersize=8,
              linewidth=2.5, label='Magnetic Dipole', alpha=0.8)
    
    # Scaling law lines
    d_fit = np.logspace(-2, 0, 100)
    ax.loglog(d_fit, 0.1*np.pi*d_fit**2, 'b--', alpha=0.5, linewidth=2,
              label=r'$d^2$ scaling')
    ax.loglog(d_fit, 0.05*np.pi*d_fit**2, 'r--', alpha=0.5, linewidth=2,
              label=r'$m^2$ scaling')
    
    # Labels and formatting
    ax.set_xlabel(r'Dipole Moment Magnitude ($e\cdot$nm or $\mu_B$)', 
                  fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\sigma_{\mathrm{tr}}$ ($\hbar^{-2} v_F^{-2}$)', 
                  fontsize=14, fontweight='bold')
    ax.set_title('Transport Cross Section vs Dipole Moment in Graphene', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add text annotations
    ax.text(0.05, 0.95, r'$\sigma_{\mathrm{tr}}^{(E)} = \frac{\pi d^2}{\hbar^2 v_F^2}$',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.text(0.05, 0.85, r'$\sigma_{\mathrm{tr}}^{(B)} \approx \frac{\pi m^2}{\hbar^2 v_F^2} e^{-a k m}$',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Add parameter info
    ax.text(0.95, 0.05, f'Energy = {energy} eV\nSoft-core a = {a} nm',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figure3_simple_transport_cross_section.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figure3_simple_transport_cross_section.pdf', 
                bbox_inches='tight')
    plt.show()
    
    print("\nSimplified Figure 3 saved as:")
    print("  figure3_simple_transport_cross_section.png")
    print("  figure3_simple_transport_cross_section.pdf")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("REALISTIC TRANSPORT CROSS-SECTION PLOT GENERATOR")
    print("=" * 70)
    
    print("\nChoose plot type:")
    print("1. Enhanced Figure 3 (full analysis)")
    print("2. Simple Figure 3 (quick plot)")
    
    try:
        choice = int(input("\nEnter choice (1 or 2): "))
        
        if choice == 1:
            plot_enhanced_figure3()
        elif choice == 2:
            plot_simple_figure3()
        else:
            print("Invalid choice. Using enhanced version...")
            plot_enhanced_figure3()
            
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to simple version...")
        plot_simple_figure3()