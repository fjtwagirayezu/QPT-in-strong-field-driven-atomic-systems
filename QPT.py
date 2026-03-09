# =============================================================================
#  QUANTUM ENERGY TELEPORTATION IN STRONGLY DRIVEN ATOMIC SYSTEMS
#  Microscopic two-qubit model + 6 plots (each saved separately as PDF)
# =============================================================================

import numpy as np
from scipy.linalg import expm, eigh
import matplotlib.pyplot as plt

# --------------------------- Global parameters ---------------------------

Delta = 1.0             # detuning Δ (sets the energy unit)
J = 0.12                # exchange interaction strength
omegas = np.linspace(0.05, 10.0, 160)   # drive strengths Ω (in units of Δ)
alpha = 0.2             # small rotation angle for Bob (kept in linear regime)

# --------------------------- Pauli and tensor products ---------------------------

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tp(A, B):
    """Kronecker (tensor) product A ⊗ B."""
    return np.kron(A, B)

# Pauli operators on the two-qubit space (A ⊗ B)
XI = tp(X, I)   # σ_x^A
YI = tp(Y, I)   # σ_y^A
ZI = tp(Z, I)   # σ_z^A

IX = tp(I, X)   # σ_x^B
IY = tp(I, Y)   # σ_y^B
IZ = tp(I, Z)   # σ_z^B

# Raising and lowering on each qubit
sigmap_A = tp(0.5 * (X + 1j * Y), I)
sigmam_A = sigmap_A.conj().T
sigmap_B = tp(I, 0.5 * (X + 1j * Y))
sigmam_B = sigmap_B.conj().T

# Exchange (flip-flop) interaction H_int = J(σ_+^A σ_-^B + σ_-^A σ_+^B)
H_int = J * (sigmap_A @ sigmam_B + sigmam_A @ sigmap_B)

# --------------------------- Local energy operator for QET (at B) ---------------------------

def local_energy_B_for_QET():
    """
    Local energy operator ε_B used in the QET analysis:

      ε_B = (Δ/2) σ_z^B + ½ H_int

    The classical drive term Ω σ_x^B is *excluded*, because it is energy
    pumped to/from atom B directly by the laser, not teleported from A.
    """
    detuning_part = (Delta / 2.0) * IZ
    interaction_part = 0.5 * H_int
    return detuning_part + interaction_part

eps_B = local_energy_B_for_QET()  # same operator for all Ω in this static toy model

# --------------------------- Hamiltonian and ground state ---------------------------

def total_hamiltonian(Omega):
    """
    Static effective Hamiltonian at a chosen drive strength Ω
    (in the rotating frame, RWA, with Ω_A = Ω_B = Ω):

      H_A = Δ/2 σ_z^A + Ω σ_x^A
      H_B = Δ/2 σ_z^B + Ω σ_x^B
      H   = H_A + H_B + H_int
    """
    H_A = (Delta / 2.0) * ZI + Omega * XI
    H_B = (Delta / 2.0) * IZ + Omega * IX
    return H_A + H_B + H_int

def ground_state(Omega):
    """
    Ground state |ψ_0(Ω)> of the static Hamiltonian H(Ω).
    This plays the role of the pre-measurement strong-field state at t_0.
    """
    H = total_hamiltonian(Omega)
    evals, evecs = eigh(H)
    psi = evecs[:, np.argmin(evals)]
    return psi / np.linalg.norm(psi)

# --------------------------- Alice's measurement & injected energy ---------------------------

def projectors_on_A(n_vec):
    """
    Build M_± = ½ (I ± n·σ^A) as operators on the full two-qubit space.
    n_vec is a 3-component real vector (n_x, n_y, n_z).
    """
    n_vec = np.array(n_vec, dtype=float)
    n_norm = np.linalg.norm(n_vec)
    if n_norm < 1e-15:
        raise ValueError("Measurement axis n_vec must be non-zero.")
    n_vec /= n_norm
    nx, ny, nz = n_vec

    M_plus  = 0.5 * (tp(I, I) + nx * XI + ny * YI + nz * ZI)
    M_minus = 0.5 * (tp(I, I) - nx * XI - ny * YI - nz * ZI)
    return M_plus, M_minus, n_vec

def injected_energy(Omega, n_vec):
    """
    Injected energy E_in for a local projective measurement on A along n_vec:

      E_in = Σ_± p_± <Ψ_±|H|Ψ_±> - <Ψ_0|H|Ψ_0>,

    where |Ψ_0> is the ground state of H(Ω) and |Ψ_±> are the post-measurement
    states after M_±.
    """
    H = total_hamiltonian(Omega)
    psi0 = ground_state(Omega)
    M_plus, M_minus, _ = projectors_on_A(n_vec)

    # Before measurement
    E_before = np.real(psi0.conj() @ H @ psi0)

    # Outcome probabilities
    p_plus = np.real(psi0.conj() @ M_plus @ psi0)
    p_plus = min(max(p_plus, 0.0), 1.0)
    p_minus = 1.0 - p_plus

    E_after_avg = 0.0

    # Outcome "+" branch
    if p_plus > 1e-14:
        psi_plus = (M_plus @ psi0) / np.sqrt(p_plus)
        E_after_avg += p_plus * np.real(psi_plus.conj() @ H @ psi_plus)

    # Outcome "-" branch
    if p_minus > 1e-14:
        psi_minus = (M_minus @ psi0) / np.sqrt(p_minus)
        E_after_avg += p_minus * np.real(psi_minus.conj() @ H @ psi_minus)

    E_in = E_after_avg - E_before
    # Numerical safety: passivity implies E_in ≥ 0
    if E_in < 0 and abs(E_in) < 1e-10:
        E_in = 0.0

    return E_in

# --------------------------- Bob's energy extraction ---------------------------

def bob_extracts_energy(psi_pm, eps_B):
    """
    Given a post-measurement state |ψ_±>, Bob performs a small rotation
    on qubit B to extract energy from ε_B.

    Linear-response logic:
      - Compute Im <ε_B σ_k^B> for k = x, y, z
      - Choose rotation axis m along that imaginary vector
      - Apply U_B = exp(-i α m·σ^B) with small α
      - Extracted energy = ⟨ε_B⟩_before - ⟨ε_B⟩_after
    """
    # Components of <ε_B σ_k^B>
    corr = np.zeros(3, dtype=complex)
    corr[0] = psi_pm.conj() @ (eps_B @ IX @ psi_pm)
    corr[1] = psi_pm.conj() @ (eps_B @ IY @ psi_pm)
    corr[2] = psi_pm.conj() @ (eps_B @ IZ @ psi_pm)

    im_corr = np.imag(corr)
    norm = np.linalg.norm(im_corr)

    if norm < 1e-14:
        # No useful imaginary component -> no QET energy
        return 0.0

    # Optimal rotation axis on B in this linearized scheme
    m = im_corr / norm
    G_B = m[0] * X + m[1] * Y + m[2] * Z

    # Local unitary on B and full two-qubit operator
    U_B = expm(-1j * alpha * G_B)
    U_full = tp(I, U_B)

    # Energies before and after Bob's operation
    E_before = np.real(psi_pm.conj() @ eps_B @ psi_pm)
    phi_pm   = U_full @ psi_pm
    E_after  = np.real(phi_pm.conj() @ eps_B @ phi_pm)

    return E_before - E_after   # positive = energy successfully extracted

# --------------------------- QET for a given Ω and measurement axis ---------------------------

def E_out_for_Omega(Omega, n_vec):
    """
    Compute teleported energy E_out(Ω) for a given drive Ω and
    a chosen measurement axis n on A:

        M_± = ½ (I ± n · σ^A)
    """
    psi0 = ground_state(Omega)
    M_plus, M_minus, _ = projectors_on_A(n_vec)

    # Outcome probabilities
    p_plus = np.real(psi0.conj() @ M_plus @ psi0)
    p_plus = min(max(p_plus, 0.0), 1.0)
    p_minus = 1.0 - p_plus

    E_out = 0.0

    # Outcome "+" branch
    if p_plus > 1e-14:
        psi_plus = (M_plus @ psi0) / np.sqrt(p_plus)
        E_out += p_plus * bob_extracts_energy(psi_plus, eps_B)

    # Outcome "-" branch
    if p_minus > 1e-14:
        psi_minus = (M_minus @ psi0) / np.sqrt(p_minus)
        E_out += p_minus * bob_extracts_energy(psi_minus, eps_B)

    return E_out

# --------------------------- Correlation measure ---------------------------

def correlation_for_Omega(Omega):
    """
    Return |<σ_+^A σ_-^B>| in the ground state at drive Ω.
    This is the key exchange-like correlation underlying QET.
    """
    psi0 = ground_state(Omega)
    C = psi0.conj() @ (sigmap_A @ sigmam_B) @ psi0
    return np.abs(C)

# --------------------------- Main scan over Ω ---------------------------

E_out_list = []
P_out_list = []
corr_vals = []
f_rep_list = []

# Fix Alice's measurement axis: along x on A (n = x̂)
n_measure = (1.0, 0.0, 0.0)

for Omega in omegas:
    # teleported energy per shot
    E_out = E_out_for_Omega(Omega, n_measure)
    E_out_list.append(E_out)

    # generalized Rabi frequency and repetition rate
    Omega_eff = np.sqrt(Delta**2 + Omega**2)
    f_rep = Omega_eff / (2.0 * np.pi)
    f_rep_list.append(f_rep)

    P_out_list.append(f_rep * E_out)

    # correlation resource
    corr_vals.append(correlation_for_Omega(Omega))

E_out_list = np.array(E_out_list)
P_out_list = np.array(P_out_list)
f_rep_list = np.array(f_rep_list)
corr_vals = np.array(corr_vals)
corr_norm = corr_vals / np.max(corr_vals) if np.max(corr_vals) > 0 else corr_vals

# --------------------------- Efficiency vs θ at Ω = Δ ---------------------------

Omega_efficiency = Delta
theta_vals = np.linspace(0.0, np.pi, 80)
efficiency_vals = []

for theta in theta_vals:
    # measurement axis n(θ) in x-z plane: n = (sinθ, 0, cosθ)
    n_vec_theta = (np.sin(theta), 0.0, np.cos(theta))
    E_out_theta = E_out_for_Omega(Omega_efficiency, n_vec_theta)
    E_in_theta = injected_energy(Omega_efficiency, n_vec_theta)
    if E_in_theta > 1e-12:
        efficiency_vals.append(abs(E_out_theta) / E_in_theta)
    else:
        efficiency_vals.append(0.0)

efficiency_vals = np.array(efficiency_vals)

# --------------------------- Phenomenological decoherence robustness ---------------------------

gammas = [0.0, 0.2, 0.5, 1.0]  # dimensionless decoherence rates
damped_P = {}

for g in gammas:
    # simple phenomenological exponential damping by γ/(f_rep + 0.1)
    damped_P[g] = P_out_list * np.exp(-g / (f_rep_list + 0.1))

# --------------------------- Qualitative power landscape vs (Ω, Δ) ---------------------------

O_vals = np.linspace(0.1, 10.0, 100)
D_vals = np.linspace(0.1, 4.0, 80)
O_grid, D_grid = np.meshgrid(O_vals, D_vals)

# Toy model: negative sign = power extracted; magnitude grows with Ω and moderate Δ
P_map = -(J * (O_grid**2 / (O_grid**2 + 2.0 * D_grid**2))) * \
        (np.sqrt(D_grid**2 + O_grid**2) / (2.0 * np.pi))

# =============================================================================
#                       GLOBAL MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({
    "axes.labelsize": 13,
    "font.size": 13,
    "legend.fontsize": 11,
    "axes.titlesize": 14
})

# =============================================================================
#                       6-PANEL SUMMARY FIGURE
# =============================================================================

fig, axs = plt.subplots(3, 2, figsize=(14, 18))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# (1) Teleported energy per shot
axs[0, 0].plot(omegas, E_out_list, lw=2, label=r"$E_{\rm out}$ per shot")
axs[0, 0].axvline(Delta, color="gray", linestyle="--", linewidth=1.5, label=r"$\Omega = \Delta$")
axs[0, 0].set_xlabel(r"Driving strength $\Omega$ (units of $\Delta$)")
axs[0, 0].set_ylabel(r"Teleported energy $E_{\rm out}$")
axs[0, 0].set_title("QET energy per operation vs driving strength")
axs[0, 0].grid(alpha=0.3)
axs[0, 0].legend()

# (2) Teleported power
axs[0, 1].plot(omegas, P_out_list, lw=2, label=r"$P_{\rm out}$")
axs[0, 1].axvline(Delta, color="gray", linestyle="--", linewidth=1.5)
axs[0, 1].set_xlabel(r"Driving strength $\Omega$ (units of $\Delta$)")
axs[0, 1].set_ylabel(r"Teleported power $P_{\rm out}$")
axs[0, 1].set_title(r"QET power  $P_{\rm out} = f_{\rm rep} E_{\rm out}$")
axs[0, 1].grid(alpha=0.3)
axs[0, 1].legend()

# (3) Quantum correlation resource
axs[1, 0].plot(omegas, corr_norm, lw=2)
axs[1, 0].set_xlabel(r"Driving strength $\Omega$ (units of $\Delta$)")
axs[1, 0].set_ylabel(r"Normalized $|\langle\sigma_+^A \sigma_-^B\rangle|$")
axs[1, 0].set_title("Quantum correlation resource vs driving strength")
axs[1, 0].grid(alpha=0.3)

# (4) Efficiency vs measurement angle
axs[1, 1].plot(theta_vals, efficiency_vals, lw=2)
axs[1, 1].set_xlabel(r"Measurement angle $\theta$ (rad)")
axs[1, 1].set_ylabel(r"Efficiency $\eta = |E_{\rm out}|/E_{\rm in}$")
axs[1, 1].set_title(r"QET efficiency vs measurement angle ($\Omega = \Delta$)")
axs[1, 1].set_xticks([0, np.pi/2, np.pi])
axs[1, 1].set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
axs[1, 1].grid(alpha=0.3)

# (5) Robustness against decoherence
for g in gammas:
    axs[2, 0].plot(omegas, damped_P[g], lw=2, label=fr"$\gamma = {g}$")
axs[2, 0].set_xlabel(r"Driving strength $\Omega$ (units of $\Delta$)")
axs[2, 0].set_ylabel(r"Damped power $P_{\rm out}$")
axs[2, 0].set_title("Phenomenological robustness against decoherence")
axs[2, 0].grid(alpha=0.3)
axs[2, 0].legend()

# (6) Power extraction landscape
im = axs[2, 1].contourf(O_grid, D_grid, P_map, levels=50, cmap="RdBu_r")
cbar = fig.colorbar(im, ax=axs[2, 1])
cbar.set_label(r"Approx. power (negative = extraction)")
axs[2, 1].set_xlabel(r"Rabi frequency $\Omega$")
axs[2, 1].set_ylabel(r"Detuning $\Delta$")
axs[2, 1].set_title("Qualitative power extraction landscape")

plt.tight_layout()
fig.savefig("QET_6panel_full.pdf", bbox_inches="tight")
plt.close(fig)

# =============================================================================
#                    INDIVIDUAL FIGURES (ONE PLOT PER PDF)
# =============================================================================

# 1. E_out vs Ω
fig1 = plt.figure(figsize=(6, 4))
ax1 = fig1.add_subplot(111)
ax1.plot(omegas, E_out_list, lw=3, label=r"$E_{\rm out}$ per shot")
ax1.axvline(Delta, color="gray", linestyle="--", linewidth=1.5, label=r"$\Omega = \Delta$")
ax1.set_xlabel(r"Driving strength $\Omega$ (units of $\Delta$)")
ax1.set_ylabel(r"Teleported energy $E_{\rm out}$")
ax1.set_title("QET energy per operation vs driving strength")
ax1.grid(alpha=0.3)
ax1.legend()
plt.tight_layout()
fig1.savefig("fig_Eout.pdf", bbox_inches="tight")
plt.show(fig1)

# 2. P_out vs Ω
fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(111)
ax2.plot(omegas, P_out_list, lw=3, label=r"$P_{\rm out}$")
ax2.axvline(Delta, color="gray", linestyle="--", linewidth=1.5, label=r"$\Omega = \Delta$")
ax2.set_xlabel(r"Driving strength $\Omega$ (units of $\Delta$)")
ax2.set_ylabel(r"Teleported power $P_{\rm out}$")
ax2.set_title(r"QET power  $P_{\rm out} = f_{\rm rep} E_{\rm out}$")
ax2.grid(alpha=0.3)
ax2.legend()
plt.tight_layout()
fig2.savefig("fig_Pout.pdf", bbox_inches="tight")
plt.show(fig2)

# 3. Correlation vs Ω
fig3 = plt.figure(figsize=(6, 4))
ax3 = fig3.add_subplot(111)
ax3.plot(omegas, corr_norm, lw=3)
ax3.set_xlabel(r"Driving strength $\Omega$ (units of $\Delta$)")
ax3.set_ylabel(r"Normalized $|\langle\sigma_+^A \sigma_-^B\rangle|$")
ax3.set_title("Quantum correlation resource vs driving strength")
ax3.grid(alpha=0.3)
plt.tight_layout()
fig3.savefig("fig_correlation.pdf", bbox_inches="tight")
plt.show(fig3)

# 4. Efficiency vs θ
fig4 = plt.figure(figsize=(6, 4))
ax4 = fig4.add_subplot(111)
ax4.plot(theta_vals, efficiency_vals, lw=3)
ax4.set_xlabel(r"Measurement angle $\theta$ (rad)")
ax4.set_ylabel(r"Efficiency $\eta = |E_{\rm out}|/E_{\rm in}$")
ax4.set_title(r"QET efficiency vs measurement angle ($\Omega = \Delta$)")
ax4.set_xticks([0, np.pi/2, np.pi])
ax4.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
ax4.grid(alpha=0.3)
plt.tight_layout()
fig4.savefig("fig_efficiency.pdf", bbox_inches="tight")
plt.show(fig4)

# 5. Decoherence robustness
fig5 = plt.figure(figsize=(6, 4))
ax5 = fig5.add_subplot(111)
for g in gammas:
    ax5.plot(omegas, damped_P[g], lw=3, label=fr"$\gamma = {g}$")
ax5.set_xlabel(r"Driving strength $\Omega$ (units of $\Delta$)")
ax5.set_ylabel(r"Damped power $P_{\rm out}$")
ax5.set_title("Robustness against decoherence")
ax5.grid(alpha=0.3)
ax5.legend()
plt.tight_layout()
fig5.savefig("fig_decoherence.pdf", bbox_inches="tight")
plt.show(fig5)

# 6. Power landscape
fig6 = plt.figure(figsize=(6.5, 4.5))
ax6 = fig6.add_subplot(111)
im2 = ax6.contourf(O_grid, D_grid, P_map, levels=50, cmap="RdBu_r")
cb2 = fig6.colorbar(im2)
cb2.set_label(r"Approx. power (negative = extraction)")
ax6.set_xlabel(r"Rabi frequency $\Omega$")
ax6.set_ylabel(r"Detuning $\Delta$")
ax6.set_title("Qualitative power extraction landscape")
plt.tight_layout()
fig6.savefig("fig_landscape.pdf", bbox_inches="tight")
plt.show(fig6)

# ---------------- Print representative values ---------------------------

i_weak = 5     # fairly weak drive
i_strong = -5  # fairly strong drive

print("Representative values:")
print(f"Weak driving  (Ω ≈ {omegas[i_weak]:.2f} Δ): "
      f"E_out ≈ {E_out_list[i_weak]:.6e},  P_out ≈ {P_out_list[i_weak]:.6e}")
print(f"Strong driving (Ω ≈ {omegas[i_strong]:.2f} Δ): "
      f"E_out ≈ {E_out_list[i_strong]:.6e},  P_out ≈ {P_out_list[i_strong]:.6e}")
print("Interpretation:")
print(" - E_out(Ω) stays of order set by J and correlations (no linear growth ~Ω).")
print(" - P_out(Ω) grows mainly because the repetition rate f_rep ∼ √(Δ²+Ω²) increases with Ω.")
