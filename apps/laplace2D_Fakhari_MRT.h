// This header defines a two‑distribution lattice Boltzmann model for
// multiphase flows using a conservative phase‑field formulation and a
// multiple‑relaxation‑time (MRT) collision operator.  The original
// implementation suffered from spurious momentum injection arising
// from a mismatch between the ordering of the discrete velocity set
// (D2Q9) used throughout the code and the ordering assumed by the
// MRT moment transformation matrices.  In the custom ordering used
// here, k=4 corresponds to the rest population, and the surrounding
// directions are enumerated in clockwise order starting from the
// negative x axis.  The MRT transformation matrices provided in the
// literature, however, assume the conventional D2Q9 ordering in which
// index 0 is the rest population and indices 1–4 correspond to the
// cardinal directions (east, north, west, south).  Without correcting
// for this permutation, spurious forces arise in the calculation of
// the viscous stress and in the MRT collision operator.  The
// modifications below introduce explicit maps between the custom and
// standard orderings and reorder the distribution functions whenever
// moment transforms are applied.

#pragma once

// ─────────────────────────   standard headers   ────────────────────────────
#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <execution>
#include <functional>
#include <chrono>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <cstdlib>
#include <numeric>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>

using namespace std;
using namespace std::chrono;

// ─────────────────────────   cell‑level helpers   ───────────────────────────
enum class CellType_laplace2D_MRT : uint8_t { bounce_back, bulk };
using  CellData = double;

// ─────────────────────────   D2Q9 constants   ───────────────────────────────
/**
 * Return the discrete velocity set, the opposite direction lookup, and the
 * weight coefficients for the D2Q9 lattice.  Note that the order of the
 * velocities follows the convention (e0,...,e8) used in Fakhari et al.,
 * Phys. Rev. E 96, 053301 (2017): k=4 corresponds to the rest
 * population.  The array indices are chosen such that opp[k] gives
 * the index of the direction opposite to k.  The nonstandard
 * ordering used here differs from the conventional D2Q9 ordering
 * assumed by the MRT moment matrices; see the LBM_laplace2D_MRT
 * struct for maps that reconcile the two.
 */
inline auto d2q9_constants_laplace2D_MRT()
{
    // k=4 is the rest (center).
    vector<array<int,2>> c_vect = {
        {-1,  0}, { 0, -1}, {-1, -1}, {-1,  1}, { 0,  0},
        { 1,  0}, { 0,  1}, { 1,  1}, { 1, -1}
    };
    vector<int>    opp_vect = { 5, 6, 7, 8, 4, 0, 1, 2, 3 };
    vector<double> t_vect   = {
        1./9., 1./9., 1./36., 1./36., 4./9.,
        1./9., 1./9., 1./36., 1./36.
    };
    return make_tuple(c_vect, opp_vect, t_vect);
}

// ─────────────────────────   geometry helpers   ────────────────────────────
struct Dim_laplace2D_MRT {
    Dim_laplace2D_MRT(int nx_, int ny_)
    : nx(nx_), ny(ny_),
      nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)),
      npop (9 * nelem) {}
    int    nx, ny;
    size_t nelem, npop;
};

// ─────────────────────────   lattice‑unit parameters   ─────────────────────
inline auto lbParameters_laplace2D_MRT(double ulb, int lref, double mu_l, double Sigma, double rho_l, double rho_g, double M)
{
    double Re   = rho_l * ulb * static_cast<double>(lref) / mu_l;
    double At   = (rho_l - rho_g ) / (rho_l + rho_g);
    double Ca   = mu_l * ulb / Sigma;
    double Pe   = ulb * static_cast<double>(lref) / M;
    double dx    = 1. / static_cast<double>(lref);
    double dt    = dx * ulb;
    return make_tuple(Re, At, Ca, Pe, dx,dt);
}

inline void printParameters_laplace2D_MRT(const Dim_laplace2D_MRT &dim,
                                              double Re, double At, double Ca, double Pe, double ulb,
                                              int N, double max_t)
{
    cout << "Laplace 2-D (MRT implementation)\n"
         << "N      = " << N        << '\n'
         << "nx     = " << dim.nx   << '\n'
         << "ny     = " << dim.ny   << '\n'
         << "Re     = " << Re       << '\n'
         << "At     = " << At       << '\n'
         << "Ca     = " << Ca       << '\n'
         << "Pe     = " << Pe       << '\n'
         << "ulb    = " << ulb      << '\n'
         << "max_t  = " << max_t    << '\n';
}

// ─────────────────────────   timing helpers   ──────────────────────────────
inline auto restartClock_laplace2D_MRT()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_laplace2D_MRT(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    int dure = us.count();
    cout << "Runtime: " << dure/1000000. << " seconds " << endl;
    cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

// ─────────────────────────   LBM functor   ─────────────────────────────────
/**
 * The LBM_laplace2D_MRT struct encapsulates the per‑cell operations for
 * advancing a multiphase LBM with a conservative phase‑field formulation.
 * A double population approach is used: f (or h) solves the phase field
 * equation with a BGK collision operator, while g solves the
 * hydrodynamic equations with a MRT collision operator.  Streaming
 * uses a standard pull scheme via the `stream` method.
 */
struct LBM_laplace2D_MRT
{
    using  CellData = double;

    // Return the number of scalar entries needed to store all populations
    // (two distribution functions f and g) and two sets of input/output
    // buffers for ping‑pong.  The factor of 2 corresponds to two
    // time–sliced copies (fin/fout and gin/gout).
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem + 2 * 9 * nelem; }

    // raw buffers
    CellData*      lattice = nullptr;  // contiguous storage for f and g
    CellType_laplace2D_MRT* flag = nullptr;  // cell types
    int*           parity = nullptr;  // toggle between fin/fout and gin/gout
    // discrete velocity set and weights (pointers into static arrays)
    std::array<int,2>* c = nullptr;
    int*           opp = nullptr;
    double*        t = nullptr;

    // model parameters
    double phi_l;
    double phi_g;         // heavy and light phase field values (e.g. 1 and 0)
    double rho_l;
    double rho_g;         // heavy and light densities
    double mu_l;          // viscosity of heavy phase
    double mu_g;          // viscosity of light phase
    double Sigma;         // surface tension coefficient
    double Fbx;           // body force in x
    double Fby;           // body force in y
    double xi;            // interface thickness
    double mobility;      // mobility parameter M used in phase field equation
    Dim_laplace2D_MRT dim;

    // constants
    static constexpr double cs2 = 1.0/3.0; // square of lattice speed of sound

    // ───────── index helpers ─────────
    size_t xyz_to_i (int x, int y) const {
        return (y + dim.ny * x);
    };

    auto i_to_xyz (int i) const {
        int iX = i / (dim.ny);
        int remainder = i % (dim.ny);
        int iY = remainder;
        return std::make_tuple(iX, iY);
    };

    // ───────── MRT transformation matrices ─────────
    // These are constant 9×9 matrices used to transform between velocity
    // space and moment space.  M is the forward transformation, M_inv is
    // its inverse.  They are taken from d'Humières et al. as used in
    // multi‑relaxation time LBM formulations.  Values are provided as
    // doubles for simplicity.  See Eq. (27) and the discussion around
    // Eq. (28) in Phys. Rev. E 96, 053301 (2017).
    static constexpr double M[9][9] = {
        { 1,  1,  1,  1,  1,  1,  1,  1,  1 },
        {-4, -1, -1, -1, -1,  2,  2,  2,  2 },
        { 4, -2, -2, -2, -2,  1,  1,  1,  1 },
        { 0,  1,  0, -1,  0,  1, -1, -1,  1 },
        { 0, -2,  0,  2,  0,  1, -1, -1,  1 },
        { 0,  0,  1,  0, -1,  1,  1, -1, -1 },
        { 0,  0, -2,  0,  2,  1,  1, -1, -1 },
        { 0,  1, -1,  1, -1,  0,  0,  0,  0 },
        { 0,  0,  0,  0,  0,  1, -1,  1, -1 }
    };
    static constexpr double M_inv[9][9] = {
        { 1./9., -1./9.,  1./9.,  0.,     0.,       0.,      0.,      0.,   0. },
        { 1./9., -1./36., -1./18., 1./6., -1./6.,     0.,      0.,     1./4., 0. },
        { 1./9., -1./36., -1./18.,  0.,     0.,      1./6.,  -1./6.,  -1./4., 0. },
        { 1./9., -1./36., -1./18.,-1./6.,  1./6.,     0.,      0.,     1./4., 0. },
        { 1./9., -1./36., -1./18.,  0.,     0.,     -1./6.,   1./6.,  -1./4., 0. },
        { 1./9.,  1./18.,  1./36.,  1./6.,  1./12.,  1./6.,    1./12.,  0.,    1./4. },
        { 1./9.,  1./18.,  1./36., -1./6., -1./12.,  1./6.,    1./12.,  0.,   -1./4. },
        { 1./9.,  1./18.,  1./36., -1./6., -1./12., -1./6.,   -1./12.,  0.,    1./4. },
        { 1./9.,  1./18.,  1./36.,  1./6.,  1./12., -1./6.,   -1./12.,  0.,   -1./4. }
    };

    /**
     * Mapping between the custom lattice ordering used in this implementation
     * and the standard D2Q9 ordering assumed by the MRT matrices.  The
     * conservative phase‑field model adopted here follows the convention
     * (e0,...,e8) described in Fakhari et al., Phys. Rev. E 96, 053301 (2017),
     * where k=4 corresponds to the rest population and the surrounding
     * directions are enumerated in clockwise order starting from the
     * negative x axis.  The MRT transformation matrices above assume the
     * conventional D2Q9 ordering in which index 0 is the rest population,
     * indices 1–4 correspond to the cardinal directions (east, north,
     * west, south), and indices 5–8 correspond to the diagonal
     * directions (northeast, northwest, southwest, southeast).  To ensure
     * that moment transforms are applied consistently, we define lookup
     * tables to convert between the custom ordering (k) and the standard
     * ordering (s).  custom_to_std[k] gives the standard index s for the
     * custom index k, while std_to_custom[s] gives the custom index k for
     * the standard index s.  These tables are used to reorder the
     * distribution functions prior to applying M and M_inv, and to map
     * the transformed distributions back to the custom ordering.
     */
    static constexpr int custom_to_std[9] = {3, 4, 7, 6, 0, 1, 2, 5, 8};
    static constexpr int std_to_custom[9]  = {4, 5, 6, 0, 1, 7, 3, 2, 8};

    // ───────── array accessors (ping‑pong) ───────
    // The lattice array stores two complete copies of the f and g
    // distributions.  The first half (0 ≤ k < 9) is f_in for the active
    // parity, the next half is f_out for the inactive parity.  The second
    // set (offset by 2*npop) stores g_in/g_out.  parity==0 corresponds
    // to reading from fin/ gin and writing to fout/ gout; parity==1
    // swaps the roles.
    inline double& f   (int i,int k)  { return lattice[              k*dim.nelem + i]; }
    inline double& fin (int i,int k)  { return lattice[*parity  *dim.npop + k*dim.nelem + i]; }
    inline double& fout(int i,int k)  { return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }

    inline double& g   (int i,int k)  { return lattice[2*dim.npop                  + k*dim.nelem + i]; }
    inline double& gin (int i,int k)  { return lattice[*parity  *dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    inline double& gout(int i,int k)  { return lattice[(1-*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    // ───────────────────────── initialization (two‑pass; uses precomputed φ₀, ∇φ₀) ──────────────────────────────
    /**
     * Initialize the distribution functions for a droplet of radius R0
     * centered in the computational domain.  The phase field is
     * initialised as a hyperbolic tangent, and the hydrodynamic
     * distribution starts from rest.  Both fin and gin buffers are
     * populated.  Note that the f0 argument refers to the f array via
     * pointer arithmetic; its address encodes the cell index i.
     */
    void iniLattice(double& f0)
    {
        auto i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz(static_cast<int>(i));

        // --- droplet geometry (circle) ---
        double xc = 0.5 * double(dim.nx);   // domain center x (cell‑centered coords)
        double yc = 0.5 * double(dim.ny);   // domain center y
        double R0 = 20.0;                   // desired radius (LU)

        // cell‑center coordinates
        double X = double(iX) + 0.5;
        double Y = double(iY) + 0.5;

        // periodic minimal‑image shift to center
        double dx = X - xc;
        double dy = Y - yc;
        double delta = sqrt(dx*dx + dy*dy) - R0;

        double phi0 = 0.5*(phi_l + phi_g);
        double amp  = 0.5*(phi_l - phi_g);
        double phi  = phi0 + amp * std::tanh( (R0 - std::hypot(dx,dy)) / (0.5*xi) );

        // populate both distribution sets with equilibria at U=0
        for (int k = 0; k < 9; ++k)
        {
            // equilibrium for phase field: h_eq = phi * w_k (1) (u=0)
            fin(i,k) = phi * t[k];
            // hydrodynamic distribution starts from rest and zero pressure
            gin(i,k) = 0.0;
        }
    };


// ───────────────────────── kernels ─────────────────────────

    // macro‑vars φ, p⋆ (normalized pressure)
    /**
     * Compute the local phase field (ϕ) and the normalized pressure p⋆
     * at the current cell.  The zeroth moment of the phase‑field
     * distribution yields ϕ and the zeroth moment of the g distribution
     * yields p⋆ (see Eqs. (12) and (32a) of the article).  Using
     * intermediate sums for opposite direction pairs improves vector
     * performance.
     */
    auto macro_phi_P(double& f0) {   // Eq. (12) , Eq. (32a)
        int i = &f0 - lattice;
        double phi , P ;

        double Xf_M1 = fin(i, 0) + fin(i, 2) + fin(i, 3);
        double Xf_P1 = fin(i, 5) + fin(i, 7) + fin(i, 8);
        double Xf_0  = fin(i, 6) + fin(i, 1) + fin(i, 4);

        double Xg_M1 = gin(i, 0) + gin(i, 2) + gin(i, 3);
        double Xg_P1 = gin(i, 5) + gin(i, 7) + gin(i, 8);
        double Xg_0  = gin(i, 6) + gin(i, 1) + gin(i, 4);

        phi  = Xf_M1 + Xf_P1 + Xf_0;
        P    = Xg_M1 + Xg_P1 + Xg_0;
        return make_tuple(phi, P);
    }

    /**
     * Compute the first moment of the g distribution to obtain the
     * momentum (u) at the current cell.  The phase‑field distribution
     * does not contribute to momentum because it tracks the phase only.
     */
    auto macro_u(double& f0)  {   // Eq. (32b) first term (moment of g)
        int i = &f0 - lattice;
        array<double,2> u;

        double Xg_M1 = gin(i, 0) + gin(i, 2) + gin(i, 3);
        double Xg_P1 = gin(i, 5) + gin(i, 7) + gin(i, 8);

        double Yg_P1 = gin(i, 3) + gin(i, 7) + gin(i, 6);
        double Yg_M1 = gin(i, 2) + gin(i, 1) + gin(i, 8);

        u[0] = Xg_P1 - Xg_M1;
        u[1] = Yg_P1 - Yg_M1;
        return u;
    }

    /**
     * Linear interpolation of density between the heavy and light phases
     * based on the phase field ϕ (Eq. (13)).  Note that phi_g and phi_l
     * represent the phase field values in the light and heavy phases
     * respectively, while rho_g and rho_l represent the corresponding
     * densities.
     */
    double local_rho(double& f0) {   // Eq. (13)
        auto [phi, P] = macro_phi_P(f0);
        return rho_g + (phi - phi_g) * (rho_l - rho_g);
    }

    /**
     * Linear interpolation of dynamic viscosity (Eq. (24)) followed by
     * computation of the hydrodynamic relaxation time (Eq. (25)).  The
     * effective viscosity is μ = μ_g + (ϕ-ϕ_g)(μ_l - μ_g).  Then
     * τ = μ/(ρ c_s^2).
     */
    double local_tau(double& f0) {   // Eq. (25)
        double rho = local_rho(f0);
        auto [phi , p] = macro_phi_P(f0);
        double mu = mu_g + (phi - phi_g) * (mu_l - mu_g);
        return mu / (rho * cs2);
    }

    /**
     * Compute the kinematic viscosity ν = τ c_s^2 (Eq. (21)).  This is
     * provided for completeness although not directly used here.
     */
    double local_nu(double& f0) {    // Eq. (21)
        auto tau = local_tau(f0);
        return tau * cs2;
    }

    /**
     * Compute the chemical potential term μϕ (Eq. (5) in the article).
     * It involves the gradient and Laplacian of the phase field and a
     * double‑well potential.  The coefficient β = 12 Σ/ξ and κ = 3 Σ ξ / 2.
     */
    double mu_phi(double& f0) {    // Eq. (5)
        auto [phi , p] = macro_phi_P(f0);
        double beta = 12.0 * Sigma / xi;
        double kappa = 3. * Sigma * xi / 2.0;
        double phi0 = (phi_l + phi_g)/2.0;
        double lap_phi = laplacian_phi(f0);
        return 4.0 * beta * (phi - phi_g) * (phi - phi_l) * (phi - phi0) - kappa * lap_phi;
    }

    // ------------------------------------------------------------------
    // ∇φ   (lattice Gradient of the phase field)
    // ------------------------------------------------------------------
    /**
     * Compute the gradient of the phase field using the isotropic
     * finite‑difference stencil described in Eq. (34).  Periodic
     * boundaries are assumed.  At bounce‑back nodes we approximate
     * the gradient using the opposite node (mirroring across the
     * boundary).  The factor 1/cs^2 appears in the denominator.
     */
    std::array<double,2> gradient_phi(double& f0){  // Eq. (34)
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(int(i));
        double gx=0.0, gy=0.0;
        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            iy = (iy + dim.ny) % dim.ny;
            int nb = xyz_to_i(ix,iy);

            double phi_nb;
            if (flag[nb] == CellType_laplace2D_MRT::bounce_back) {
                // mirror for bounce‑back boundary: use the cell in the opposite direction
                int ixbb = iX - c[k][0];
                int iybb = iY - c[k][1];
                ixbb = (ixbb + dim.nx) % dim.nx;
                iybb = (iybb + dim.ny) % dim.ny;
                int nbb = xyz_to_i(ixbb,iybb);
                auto [phi_mirror, P_nb] = macro_phi_P(lattice[nbb]);
                phi_nb = phi_mirror;
            }
            else {
                auto [phi_nbb, P_nbb] = macro_phi_P(lattice[nb]);
                phi_nb = phi_nbb;
            }
            gx += t[k] * c[k][0] * phi_nb;
            gy += t[k] * c[k][1] * phi_nb;
        }
        return { gx / cs2, gy / cs2 };
    }

    // ------------------------------------------------------------------
    // ∇²φ   (lattice Laplacian of the local density phi)
    // ------------------------------------------------------------------
    /**
     * Compute the Laplacian of the phase field using Eq. (35).
     */
    double laplacian_phi(double& f0){   // Eq. (35)
        size_t idx   = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(int(idx));
        auto [phi_c, p] = macro_phi_P(lattice[idx]);
        double sum   = 0.0;
        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            iy = (iy + dim.ny) % dim.ny;
            int nb = xyz_to_i(ix,iy);
            if (flag[nb] != CellType_laplace2D_MRT::bounce_back) {
                auto [phi_n, P_n] = macro_phi_P(lattice[nb]);
                sum += t[k] * (phi_n - phi_c);
            }
        }
        return 2.0 * sum / cs2;
    }

    // ------------------------------------------------------------------
    // ∇ρ   (lattice Gradient of the local density rho)
    // ------------------------------------------------------------------
    /**
     * Compute the gradient of density by scaling the gradient of the
     * phase field (Eq. (33)).  Since ρ = ρ_g + (ϕ-ϕ_g)(ρ_l-ρ_g), we
     * have ∇ρ = (ρ_l - ρ_g) ∇ϕ.
     */
    std::array<double,2> gradient_rho(double& f0){  // Eq. (33)
        auto grad_phi = gradient_phi(f0);
        double delta_rho = (rho_l - rho_g);
        return { delta_rho * grad_phi[0], delta_rho * grad_phi[1] };
    }

    // ------------------------------------------------------------------
    // Forces (Fs , Fb , Fp , Fμ)
    // ------------------------------------------------------------------
    /**
     * Surface tension force Fs = μϕ ∇ϕ (Eq. (4)).
     */
    std::array<double,2> Fs(double& f0){
        auto grad_phi = gradient_phi(f0);
        double mu_phi_val = mu_phi(f0);
        return { mu_phi_val * grad_phi[0], mu_phi_val * grad_phi[1] };
    }

    /**
     * Body force due to gravity or other external fields: F_b = ρ g (Eq. (??)).
     */
    std::array<double,2> Fb(double& f0){
        double rho = local_rho(f0);
        return { rho * Fbx, rho * Fby };
    }

    /**
     * Pressure gradient force F_p = −p⋆ c_s^2 ∇ρ (Eq. (19)).  The
     * normalized pressure p⋆ is returned by macro_phi_P.
     */
    std::array<double,2> Fp(double& f0){
        auto [phi , Pstar] = macro_phi_P(f0);
        auto grad_rho = gradient_rho(f0);
        return { -Pstar * cs2 * grad_rho[0], -Pstar * cs2 * grad_rho[1] };
    }

    /**
     * Viscous force F_μ computed using the MRT stress tensor (Eq. (31)).
     * This routine assembles the stress via a moment‑space multiplication
     * (M^{-1} S M) acting on g − g_eq and then contracts with ∇ρ.  The
     * custom ordering of the discrete velocities is converted to the
     * standard ordering assumed by the MRT matrices using
     * custom_to_std and std_to_custom.  Without this reordering the
     * off‑diagonal components of the stress tensor do not cancel
     * properly, resulting in spurious net forces.
     */
    std::array<double,2> Fmu(double& f0){
        auto idx = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(int(idx));
        auto [phi, pstar] = macro_phi_P(f0);
        auto u            = mid_velocity(f0);
        double rho  = local_rho(f0);
        double tau  = local_tau(f0);
        double nu   = local_nu(f0);
        auto grad_rho = gradient_rho(f0);

        // Gather g and build g_eq (without forcing correction) in the standard
        // ordering.  The custom ordering (k) is mapped to the standard
        // ordering (s) via custom_to_std.  geq is computed in custom
        // ordering and then placed into the standard array.  This
        // reordering is essential to align the distribution functions with
        // the MRT matrices M and M_inv.
        double gvec_std[9];
        double geq_std[9];
        double u_sq = (u[0]*u[0] + u[1]*u[1]) / (2.0*cs2);
        for (int k = 0; k < 9; ++k) {
            int s = custom_to_std[k];
            // current distribution
            double gk = gin(idx,k);
            // compute equilibrium in custom ordering
            double ex = c[k][0];
            double ey = c[k][1];
            double ck_u = ex*u[0] + ey*u[1];
            double gamma_k = t[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - u_sq);
            double geq_k = pstar * t[k] + (gamma_k - t[k]);
            gvec_std[s] = gk;
            geq_std[s]  = geq_k;
        }

        // Compute moment vector m = M*(g_std - g_eq_std)
        double m_vec[9];
        for (int r = 0; r < 9; ++r) {
            double sum = 0.0;
            for (int a = 0; a < 9; ++a) {
                sum += M[r][a] * (gvec_std[a] - geq_std[a]);
            }
            m_vec[r] = sum;
        }

        // Relaxation matrix S_diag (Eq. (28)), with s_ν = 1/(τ + 0.5)
        double s_nu = 1.0 / (tau + 0.5);
        double Sdiag[9] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, s_nu, s_nu };

        // Post‑collision moments s_m = S_diag * m
        double s_m[9];
        for (int r = 0; r < 9; ++r) {
            s_m[r] = Sdiag[r] * m_vec[r];
        }

        // Convert back to velocity space in standard ordering: A_std = M_inv * s_m
        double A_std[9];
        for (int a = 0; a < 9; ++a) {
            double sum = 0.0;
            for (int r = 0; r < 9; ++r) {
                sum += M_inv[a][r] * s_m[r];
            }
            A_std[a] = sum;
        }

        // Reorder A from standard to custom ordering
        double A_custom[9];
        for (int k = 0; k < 9; ++k) {
            int s = custom_to_std[k];
            A_custom[k] = A_std[s];
        }

        // Contract stress tensor with gradient of density using custom c and A_custom
        double Txx = 0.0, Txy = 0.0, Tyy = 0.0;
        for (int beta = 0; beta < 9; ++beta) {
            double ex = static_cast<double>(c[beta][0]);
            double ey = static_cast<double>(c[beta][1]);
            Txx += A_custom[beta] * ex * ex;
            Txy += A_custom[beta] * ex * ey;
            Tyy += A_custom[beta] * ey * ey;
        }
        double pref = - nu / cs2;
        double Fx = pref * ( Txx * grad_rho[0] + Txy * grad_rho[1] );
        double Fy = pref * ( Txy * grad_rho[0] + Tyy * grad_rho[1] );
        return {Fx, Fy};
    }

    /**
     * Total force excluding viscous MRT contribution: Fs + Fb + Fp.
     */
    std::array<double,2> Total_Fsbp(double& f0){
        auto F_s = Fs(f0);
        auto F_b = Fb(f0);
        auto F_p = Fp(f0);
        return { F_s[0] + F_b[0] + F_p[0] , F_s[1] + F_b[1] + F_p[1] };
    }

    /**
     * Total force including viscous MRT contribution: Fs + Fb + Fp + Fμ.
     */
    std::array<double,2> Total_F(double& f0){
        auto F_s = Fs(f0);
        auto F_b = Fb(f0);
        auto F_p = Fp(f0);
        auto F_mu = Fmu(f0);
        return { F_s[0] + F_b[0] + F_p[0] + F_mu[0] , F_s[1] + F_b[1] + F_p[1] + F_mu[1] };
    }

    // ------------------------------------------------------------------
    // Velocities including forcing terms
    // ------------------------------------------------------------------
    /**
     * Compute the mid‑point velocity used in the update of the phase
     * field (Eq. (32b) with only Fs+Fb+Fp).  Force is divided by
     * 2ρ according to the trapezoidal rule.
     */
    std::array<double,2> mid_velocity(double& f0){  // Eq. (32b)
        auto u = macro_u(f0);
        auto force = Total_Fsbp(f0);
        double rho = local_rho(f0);
        return { u[0] + force[0]/(2.0*rho), u[1] + force[1]/(2.0*rho) };
    }

    /**
     * Compute the full velocity including viscous contributions (Eq. (32b)).
     */
    std::array<double,2> velocity(double& f0){
        auto u = macro_u(f0);
        auto force = Total_F(f0);
        double rho = local_rho(f0);
        return { u[0] + force[0]/(2.0*rho), u[1] + force[1]/(2.0*rho) };
    }

    /**
     * Compute the k‑independent part of the phase‑field forcing term
     * Fϕ_α (Eq. (7)).  The actual forcing term is Fϕ_k = δt w_k [1−4(ϕ−ϕ0)^2] (e_k·∇ϕ/|∇ϕ|)/ξ.
     * Here we return the factor [1−4(ϕ−ϕ0)^2]/ξ * (∇ϕ/|∇ϕ|); the dot
     * product with c[k] and multiplication by w[k] and δt is done in
     * the collision step.
     */
    std::array<double,2> Fphi_fac(double& f0){  // Eq. (7) (only the k‑independent part)
        auto [phi , p] = macro_phi_P(f0);
        auto grad_phi = gradient_phi(f0);
        double phi0 = (phi_g + phi_l) / 2.0;
        double fac = (1.0 - 4.0*std::pow(phi - phi0, 2)) / xi;
        double s = std::sqrt(grad_phi[0] * grad_phi[0] + grad_phi[1] * grad_phi[1]) + 1e-30;
        return { fac * grad_phi[0] / s, fac * grad_phi[1] / s };
    }

    // ------------------------------------------------------------------
    // Streaming with bounce‑back treatment
    // ------------------------------------------------------------------
    /**
     * Stream a single population along a discrete velocity.  If the
     * neighbor cell is flagged as bounce_back, the distribution is
     * reflected (assigned to the opposite direction of the current cell);
     * otherwise it is sent to the neighbor.  This method is used for
     * both f and g simultaneously.
     */
    void stream(int i,int k,int iX,int iY,double pf,double pg)
    {
        int XX = iX + c[k][0];
        int YY = iY + c[k][1];
        XX = (XX + dim.nx) % dim.nx;
        YY = (YY + dim.ny) % dim.ny;
        size_t nb = xyz_to_i(XX,YY);
        if (flag[nb] == CellType_laplace2D_MRT::bounce_back) {
            // bounce back: reflect into opposite direction within same cell
            fout(i,opp[k]) = pf;
            gout(i,opp[k]) = pg;
        }
        else {
            // propagate to neighbor cell
            fout(nb,k) = pf;
            gout(nb,k) = pg;
        }
    };

    // ------------------------------------------------------------------
    // Main per‑cell operator
    // ------------------------------------------------------------------
    /**
     * Perform the collision and streaming operations on a single cell.
     * For bulk cells we compute the macroscopic quantities, forces,
     * equilibria, and then update both f and g distributions.  The
     * phase field uses a BGK relaxation and the hydrodynamics use an
     * MRT relaxation.  We implement a two‑phase streaming step:
     * for non‑rest populations (k < 4) we stream both k and opp[k],
     * while the rest population (k == 4) is updated in place.
     */
    void operator()(double& f0)
    {
        int idx = &f0 - lattice;
        if (flag[idx] != CellType_laplace2D_MRT::bulk) {
            return;  // bounce‑back cells do not collide; streaming handled elsewhere
        }
        // Compute macroscopic variables
        auto  u   = velocity(f0);              // full velocity including forcing
        auto [phi, Pstar] = macro_phi_P(f0);   // phase field and pressure
        double rho = local_rho(f0);
        double tau = local_tau(f0);
        double tau_phi = mobility / cs2;       // τ_φ = M/(c_s^2) (Eq. (11))

        auto grad_phi = gradient_phi(f0);
        auto lap_phi  = laplacian_phi(f0);
        auto grad_rho = gradient_rho(f0);
        auto F_sbp = Total_Fsbp(f0);  // Fs + Fb + Fp
        auto F_mu  = Fmu(f0);
        auto F_tot = std::array<double,2>{ F_sbp[0] + F_mu[0], F_sbp[1] + F_mu[1] };
        auto F_phi = Fphi_fac(f0);
        double u_sq = (u[0]*u[0] + u[1]*u[1]) / (2.0*cs2);

        // --- Phase field BGK collision ---
        // Precompute pf_out for all directions
        double pf_out[9];
        for (int k = 0; k < 9; ++k) {
            double ex = c[k][0];
            double ey = c[k][1];
            double ck_u = ex*u[0] + ey*u[1];
            // γ_k in Eq. (10): t_k * [1 + e_k·u/cs^2 + (e_k·u)^2/(2 cs^4) − u·u/(2 cs^2)]
            double gamma_k = t[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - u_sq);
            // BGK equilibrium (without force) for phase field: h_eq = φ γ_k
            // Force term for phase field: Fϕ_k = δt w_k (F_phi·c_k)
            double Ff_k = t[k] * (F_phi[0] * ex + F_phi[1] * ey);
            double feq_bar = phi * gamma_k - Ff_k / 2.0;
            // f_post = f_in - (f_in - feq_bar)/(τ_φ + 0.5) + Fϕ_k
            double f_in_val = fin(idx,k);
            pf_out[k] = f_in_val - (f_in_val - feq_bar) / (tau_phi + 0.5) + Ff_k;
        }

        // --- Hydrodynamic MRT collision ---
        // Gather gvec and compute geq and forcing Fg in standard ordering
        double gvec_std[9];
        double geq_std[9];
        double Fg_std[9];
        for (int k = 0; k < 9; ++k) {
            int s = custom_to_std[k];
            double ex = c[k][0];
            double ey = c[k][1];
            double ck_u = ex*u[0] + ey*u[1];
            double gamma_k = t[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - u_sq);
            double g_k  = gin(idx,k);
            double geq_k= Pstar * t[k] + (gamma_k - t[k]);
            double F_k  = t[k] * (F_tot[0] * ex + F_tot[1] * ey) / (rho * cs2);
            gvec_std[s] = g_k;
            geq_std[s]  = geq_k;
            Fg_std[s]   = F_k;
        }
        // moment space arrays
        double m_vec[9] = {0.0}, m_eq[9] = {0.0}, Fm[9] = {0.0}, m_post[9] = {0.0};
        // m_vec = M * gvec_std, m_eq = M * geq_std, Fm = M * Fg_std
        for (int r = 0; r < 9; ++r) {
            double sum = 0.0, sum_eq = 0.0, sum_f = 0.0;
            for (int a = 0; a < 9; ++a) {
                sum    += M[r][a] * gvec_std[a];
                sum_eq += M[r][a] * geq_std[a];
                sum_f  += M[r][a] * Fg_std[a];
            }
            m_vec[r] = sum;
            m_eq[r]  = sum_eq;
            Fm[r]    = sum_f;
        }
        // relaxation diagonal: s_ν only applies to shear modes (indices 7 and 8)
        double s_nu = 1.0 / (tau + 0.5);
        double Sdiag[9] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, s_nu, s_nu };
        // post‑collision moments: m' = m - S (m - m_eq) + (I - S/2) * Fm
        for (int r = 0; r < 9; ++r) {
            double Sr = Sdiag[r];
            m_post[r] = m_vec[r] - Sr * (m_vec[r] - m_eq[r]) + (1.0 - 0.5 * Sr) * Fm[r];
        }
        // back to velocity space in standard ordering: g_post_std = M_inv * m_post
        double g_post_std[9];
        for (int a = 0; a < 9; ++a) {
            double sum = 0.0;
            for (int r = 0; r < 9; ++r) {
                sum += M_inv[a][r] * m_post[r];
            }
            g_post_std[a] = sum;
        }
        // Reorder g_post_std back to custom ordering
        double g_post[9];
        for (int k = 0; k < 9; ++k) {
            int s = custom_to_std[k];
            g_post[k] = g_post_std[s];
        }

        // obtain current coordinates for streaming
        auto [iX,iY] = i_to_xyz(idx);

        // Stream non‑rest populations (k<4) and their opposites
        for (int k = 0; k < 4; ++k) {
            double pf  = pf_out[k];
            double pg  = g_post[k];
            double pf_op = pf_out[ opp[k] ];
            double pg_op = g_post[ opp[k] ];
            stream(idx,     k      , iX, iY, pf   , pg);
            stream(idx, opp[k], iX, iY, pf_op, pg_op);
        }
        // Rest population k=4: update locally without streaming
        int k0 = 4;
        // BGK update for f at rest
        fout(idx,k0) = pf_out[k0];
        // MRT update for g at rest
        gout(idx,k0) = g_post[k0];
    }
};

// ───────────────────────── VTK output ─────────────────────────────────
/**
 * Write diagnostic fields (phase field, density, pressure, velocity and
 * flag) to a VTK file for visualization.
 */
void saveVtkFields_laplace2D_MRT(LBM_laplace2D_MRT& lbm, int time_iter, double dx = 0.) {
    Dim_laplace2D_MRT const& dim = lbm.dim;
    if (dx == 0.0) dx = 1.0 / dim.nx;
    int dimZ = 1;

    stringstream ss;
    ss << "sol_" << setw(7) << setfill('0') << time_iter << ".vtk";
    ofstream os(ss.str());
    os << "# vtk DataFile Version 2.0\n";
    os << "iteration " << time_iter << "\n";
    os << "ASCII\n\n";
    os << "DATASET STRUCTURED_POINTS\n";
    os << "DIMENSIONS " << dim.nx << " " << dim.ny << " " << dimZ << "\n";
    os << "ORIGIN 0 0 0\n";
    os << "SPACING " << dx << " " << dx << " " << dx << "\n\n";
    os << "POINT_DATA " << dim.nx * dim.ny * dimZ << "\n";

    // φ
    os << "SCALARS phi float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            float val = 0.f;
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                auto [phi, p] = lbm.macro_phi_P(lbm.lattice[i]);
                val = phi;
            }
            os << val << " ";
        }
        os << "\n";
    }
    os << "\n";

    // ρ
    os << "SCALARS density float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            float val = 0.f;
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                val = lbm.local_rho(lbm.lattice[i]);
            }
            os << val << " ";
        }
        os << "\n";
    }
    os << "\n";

    // p⋆ρc_s^2 (pressure)
    os << "SCALARS pressure float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double val = 0.0;
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                auto [phi, p] = lbm.macro_phi_P(lbm.lattice[i]);
                double rho = lbm.local_rho(lbm.lattice[i]);
                val = p * rho / 3.0;            }
            os << val << " ";
        }
        os << "\n";
    }
    os << "\n";

    // Velocity (vector field)
    os << "VECTORS velocity float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double ux = 0., uy = 0.;
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                auto u = lbm.velocity(lbm.lattice[i]);
                ux = u[0];
                uy = u[1];
            }
            os << ux << " " << uy << " 0\n";
        }
    }
    os << "\n";

    // Flag
    os << "SCALARS Flag int 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x, y);
            os << ((lbm.flag[i] == CellType_laplace2D_MRT::bounce_back) ? 1 : 0) << " ";
        }
        os << "\n";
    }
    os << "\n";



        // ----------------- Forces (vector fields) -----------------

    // Fs (surface tension)
    os << "VECTORS Fs float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double fx = 0., fy = 0.;
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                auto F = lbm.Fs(lbm.lattice[i]);
                fx = F[0]; fy = F[1];
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";

    // Fb (body force)
    os << "VECTORS Fb float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double fx = 0., fy = 0.;
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                auto F = lbm.Fb(lbm.lattice[i]);
                fx = F[0]; fy = F[1];
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";

    // Fp (pressure-gradient force)
    os << "VECTORS Fp float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double fx = 0., fy = 0.;
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                auto F = lbm.Fp(lbm.lattice[i]);
                fx = F[0]; fy = F[1];
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";

    // Fmu (viscous MRT contribution)
    os << "VECTORS Fmu float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double fx = 0., fy = 0.;
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                auto F = lbm.Fmu(lbm.lattice[i]);
                fx = F[0]; fy = F[1];
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";

    // Total force (Fs + Fb + Fp + Fmu)
    os << "VECTORS Ftotal float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double fx = 0., fy = 0.;
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                auto F = lbm.Total_F(lbm.lattice[i]);
                fx = F[0]; fy = F[1];
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";

    // (optional) Total_Fsbp (Fs + Fb + Fp) used in mid_velocity
    os << "VECTORS Fsbp float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double fx = 0., fy = 0.;
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                auto F = lbm.Total_Fsbp(lbm.lattice[i]);
                fx = F[0]; fy = F[1];
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";

}

// ───────────────────────── Diagnostics helpers ─────────────────────────
/**
 * Optional diagnostics to monitor global force and momentum.  When
 * compiled with the preprocessor macro `LBM_DIAGNOSTICS` defined, the
 * function `computeDiagnostics_laplace2D_MRT` will accumulate and print
 * the total viscous force (Σ_x F_μ), total momentum (Σ_x ρu), and the
 * phase‑field centre of mass at selected iterations.  These diagnostics
 * are useful for verifying that spurious forces vanish for a symmetric
 * droplet in a periodic domain.  The diagnostics are off by default to
 * avoid incurring overhead in production runs.  To enable them, add
 * `-DLBM_DIAGNOSTICS` to the compiler flags and call
 * `computeDiagnostics_laplace2D_MRT` from within the time‑stepping loop.
 */
#ifdef LBM_DIAGNOSTICS
void computeDiagnostics_laplace2D_MRT(LBM_laplace2D_MRT& lbm, int iter, double time)
{
    const auto& dim = lbm.dim;
    double sumFx = 0.0, sumFy = 0.0;
    double sumMx = 0.0, sumMy = 0.0;
    double sumPhi = 0.0, sumPx = 0.0, sumPy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                // viscous force
                auto Fmu = lbm.Fmu(lbm.lattice[i]);
                sumFx += Fmu[0];
                sumFy += Fmu[1];
                // momentum = rho * u
                double rho = lbm.local_rho(lbm.lattice[i]);
                auto u = lbm.velocity(lbm.lattice[i]);
                sumMx += rho * u[0];
                sumMy += rho * u[1];
                // centre of mass of phase field
                auto [phi, P] = lbm.macro_phi_P(lbm.lattice[i]);
                sumPhi += phi;
                // positions in lattice units (use cell centre coordinates)
                double X = (double)x + 0.5;
                double Y = (double)y + 0.5;
                sumPx += phi * X;
                sumPy += phi * Y;
            }
        }
    }
    double comX = 0.0, comY = 0.0;
    if (sumPhi != 0.0) {
        comX = sumPx / sumPhi;
        comY = sumPy / sumPhi;
    }
    std::cout << std::setprecision(8) << "[Diag] t=" << time
              << " iter=" << iter
              << "  ΣF_μ = (" << sumFx << ", " << sumFy << ")"
              << "  Σρu = (" << sumMx << ", " << sumMy << ")"
              << "  COM_φ = (" << comX << ", " << comY << ")"
              << std::endl;
}
#endif // LBM_DIAGNOSTICS

// ───────────────────────── Energy computation (use stored Ux,Uy) ──────────────────────────────
/**
 * Compute the total kinetic energy of the fluid (for diagnostics).
 */
double computeEnergy_laplace2D_MRT(LBM_laplace2D_MRT& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_laplace2D_MRT::bulk) {
                auto u = lbm.velocity(lbm.lattice[i]);
                double ux = u[0];
                double uy = u[1];
                energy += ux*ux + uy*uy;
            }
        }
    }
    return 0.5 * energy;
}

// ───────────────────────── Geometry initialization ─────────────────────────
/**
 * Initialise the geometry by setting bounce‑back flags along specified
 * walls.  Currently only a trivial example is provided; users should
 * modify this function to impose their own boundary conditions.  All
 * non‑bounce‑back cells are flagged as bulk and their distributions
 * remain unchanged.
 */
void inigeom_laplace2D_MRT(LBM_laplace2D_MRT& lbm) {
    Dim_laplace2D_MRT const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            // Example bounce‑back condition on fictitious horizontal walls
            if (y == dim.ny + 100 || y == dim.ny + 111) {
                lbm.flag[i] = CellType_laplace2D_MRT::bounce_back;
                for (int k = 0; k < 9; ++k) {
                    lbm.fin(i,k) = 0.0;
                    lbm.gin(i,k) = 0.0;
                }
            }
            else {
                lbm.flag[i] = CellType_laplace2D_MRT::bulk;
            }
        }
    }
}

// ─────────────────────────   main driver (Laplace MRT)  ────────────────────────
/**
 * Run the LBM solver for the conservative phase‑field model.  The
 * configuration file "config_laplace2D_MRT.txt" under
 * apps/Config_Files should specify simulation parameters.  The
 * implementation below closely follows the stand‑alone C++ version
 * shipped with the original problem description.
 */
void laplace2D_MRT()
{
    ifstream contfile("../apps/Config_Files/config_laplace2D_MRT.txt");
    if (!contfile.is_open()) {
        throw invalid_argument(
            "Config file not found. It should be named "
            "\"config_laplace2D_MRT.txt\" in apps/Config_Files.");
    }
    // Default parameters
    double ulb=1.0, max_t=0.01, phi_l=1.0, phi_g=0.0, rho_l=1.0, rho_g=1e-3;
    double Sigma=0.01, xi=4.0, M=0.02, R=0.0, Fbx=0.0, Fby = 0.0;
    double mu_l=0.1, mu_g=0.0001;
    int N=128, out_freq=1000, vtk_freq=1000;

    // Read parameters from file (if present)
    string line, param, value;
    while (getline(contfile, line)) {
        if (auto pos = line.find('#'); pos != string::npos) line.erase(pos);
        if (line.empty()) continue;
        istringstream ls(line);
        if (!(ls >> param >> value)) continue;
        if      (param == "N")          N       = stoi(value);
        else if (param == "max_t")      max_t   = stod(value);
        else if (param == "out_freq")   out_freq= stoi(value);
        else if (param == "vtk_freq")   vtk_freq= stoi(value);
        else if (param == "phi_l")      phi_l   = stod(value);
        else if (param == "phi_g")      phi_g   = stod(value);
        else if (param == "rho_l")      rho_l   = stod(value);
        else if (param == "rho_g")      rho_g   = stod(value);
        else if (param == "Sigma")      Sigma   = stod(value);
        else if (param == "xi")         xi       = stod(value);
        else if (param == "M")          M       = stod(value);
        else if (param == "ulb")        ulb     = stod(value);
        else if (param == "Fbx")        Fbx     = stod(value);
        else if (param == "Fby")        Fby     = stod(value);
        else if (param == "mu_l")       mu_l    = stod(value);
        else if (param == "mu_g")       mu_g    = stod(value);
        else {
            cerr << "Warning: unknown parameter \"" << param << "\"\n";
        }
    }

    // lattice parameters (omega only for logging; g uses local tau)
    Dim_laplace2D_MRT dim {N, N};
    auto [Re, At, Ca, Pe, dx,dt] = lbParameters_laplace2D_MRT(ulb, N, mu_l, Sigma, rho_l , rho_g , M);
    printParameters_laplace2D_MRT(dim, Re, At, Ca, Pe, ulb, N, max_t);

    // allocate distributions
    vector<CellData> lattice_vect(LBM_laplace2D_MRT::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();
    // flags
    vector<CellType_laplace2D_MRT> flag_vect(dim.nelem);
    CellType_laplace2D_MRT* flag = &flag_vect[0];
    // parity
    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];
    // load discrete velocities and weights
    auto [c_vect, opp_vect, t_vect] = d2q9_constants_laplace2D_MRT();
    // instantiate LBM functor
    LBM_laplace2D_MRT lbm{
        lattice, flag, parity,
        c_vect.data(), opp_vect.data(), t_vect.data(),
        phi_l, phi_g, rho_l, rho_g,
        mu_l, mu_g, Sigma,
        Fbx, Fby, xi, M,
        dim
    };

    // initialise distributions with the droplet profile
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });
    // initialise geometry (boundary conditions)
    inigeom_laplace2D_MRT(lbm);

    // timing and diagnostics
    auto [start, clock_iter] = restartClock_laplace2D_MRT();
    ofstream efile("energy.dat");

    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
            saveVtkFields_laplace2D_MRT(lbm, time_iter);
        }
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(4) << time_iter * dt << setprecision(3)
                 << " [" << time_iter * dt / max_t * 100. << "%]" << endl;
            double energy = computeEnergy_laplace2D_MRT(lbm) * dx * dx / (dt * dt);
            cout << "Average energy: " << setprecision(8) << energy << endl;
            efile << setw(10) << time_iter * dt << setw(16) << setprecision(8) << energy << endl;
#ifdef LBM_DIAGNOSTICS
            // Print diagnostics (sum of viscous force, momentum, and centre of mass)
            computeDiagnostics_laplace2D_MRT(lbm, time_iter, time_iter * dt);
#endif
        }
        // parallel collision/streaming on all cells
        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        // swap buffers
        *parity = 1 - *parity;
        ++clock_iter;
    }
    printMlups_laplace2D_MRT(start, clock_iter, dim.nelem);
}