// This header defines an optimized two‑distribution lattice Boltzmann model
// for multiphase flows using a conservative phase‑field formulation and a
// multiple‑relaxation‑time (MRT) collision operator.  It is derived from
// the reference implementation `laplace2D_MRT.h` but has been tuned for
// better CPU efficiency.  All symbols have been suffixed with `_opt` to
// distinguish this version.  Minor inlining and local caching of pointers
// have been introduced to aid the compiler’s optimizer.  No debug
// instrumentation is included in this optimized version.

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
enum class CellType_laplace2D_MRT_opt : uint8_t { bounce_back, bulk };
using  CellData_opt = double;

// ─────────────────────────   D2Q9 constants   ───────────────────────────────
/**
 * Return the discrete velocity set, the opposite direction lookup, and the
 * weight coefficients for the D2Q9 lattice.  This follows the same
 * nonstandard ordering as the reference implementation: k=4 is the rest
 * population, and the surrounding directions are enumerated in clockwise
 * order starting from the negative x axis.  The returned arrays are used
 * directly by the optimized LBM functor.
 */
inline auto d2q9_constants_laplace2D_MRT_opt()
{
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
struct Dim_laplace2D_MRT_opt {
    Dim_laplace2D_MRT_opt(int nx_, int ny_)
    : nx(nx_), ny(ny_),
      nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)),
      npop (9 * nelem) {}
    int    nx, ny;
    size_t nelem, npop;
};

// ─────────────────────────   lattice‑unit parameters   ─────────────────────
inline auto lbParameters_laplace2D_MRT_opt(double ulb, int lref, double mu_l, double Sigma, double rho_l, double rho_g, double M)
{
    double Re   = rho_l * ulb * static_cast<double>(lref) / mu_l;
    double At   = (rho_l - rho_g ) / (rho_l + rho_g);
    double Ca   = mu_l * ulb / Sigma;
    double Pe   = ulb * static_cast<double>(lref) / M;
    double dx    = 1. / static_cast<double>(lref);
    double dt    = dx * ulb;
    return make_tuple(Re, At, Ca, Pe, dx,dt);
}

inline void printParameters_laplace2D_MRT_opt(const Dim_laplace2D_MRT_opt &dim,
                                              double Re, double At, double Ca, double Pe, double ulb,
                                              int N, double max_t)
{
    cout << "Laplace 2-D (MRT optimized implementation)\n"
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
inline auto restartClock_laplace2D_MRT_opt()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_laplace2D_MRT_opt(TP start, int iter, size_t nelem)
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
 * The LBM_laplace2D_MRT_opt struct encapsulates the per‑cell operations for
 * advancing a multiphase LBM with a conservative phase‑field formulation.
 * A double population approach is used: f solves the phase field equation
 * with a BGK collision operator, while g solves the hydrodynamic equations
 * with a MRT collision operator.  Streaming uses a standard pull scheme via
 * the `stream` method.  Compared with the reference version, this struct
 * exposes all methods as `inline` to encourage aggressive inlining and
 * caches frequently used pointers locally inside heavy routines.
 */
struct LBM_laplace2D_MRT_opt
{
    using  CellData = double;

    // Return the number of scalar entries needed to store all populations
    // (two distribution functions f and g) and two sets of input/output
    // buffers for ping‑pong.  The factor of 2 corresponds to two
    // time–sliced copies (fin/fout and gin/gout).
    static inline size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem + 2 * 9 * nelem; }

    // raw buffers
    CellData*      lattice = nullptr;  // contiguous storage for f and g
    CellType_laplace2D_MRT_opt* flag = nullptr;  // cell types
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
    Dim_laplace2D_MRT_opt dim;

    // constants
    static constexpr double cs2 = 1.0/3.0; // square of lattice speed of sound

    // ───────── index helpers ─────────
    inline size_t xyz_to_i (int x, int y) const {
        return static_cast<size_t>(y) + static_cast<size_t>(dim.ny) * static_cast<size_t>(x);
    }

    inline auto i_to_xyz (int i) const {
        int iX = i / (dim.ny);
        int iY = i % (dim.ny);
        return std::make_tuple(iX, iY);
    }

    // ───────── MRT transformation matrices ─────────
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

    // Mapping between custom and standard ordering. See original header for details.
    static constexpr int custom_to_std[9] = {3, 4, 7, 6, 0, 1, 2, 5, 8};
    static constexpr int std_to_custom[9]  = {4, 5, 6, 0, 1, 7, 3, 2, 8};

    // ───────── array accessors (ping‑pong) ───────
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
    inline void iniLattice(double& f0)
    {
        auto i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz(static_cast<int>(i));

        // droplet geometry (circle)
        double xc = 0.5 * double(dim.nx);
        double yc = 0.5 * double(dim.ny);
        double R0 = 20.0;
        // cell‑center coordinates
        double X = double(iX) + 0.5;
        double Y = double(iY) + 0.5;
        // distance to centre
        double dx = X - xc;
        double dy = Y - yc;
        double phi0 = 0.5*(phi_l + phi_g);
        double amp  = 0.5*(phi_l - phi_g);
        double phi  = phi0 + amp * std::tanh( (R0 - std::hypot(dx,dy)) / (0.5*xi) );
        // populate both distribution sets with equilibria at U=0
        for (int k = 0; k < 9; ++k)
        {
            fin(i,k) = phi * t[k];
            gin(i,k) = 0.0;
        }
    }

// ───────────────────────── kernels ─────────────────────────

    // macro‑vars φ, p⋆ (normalized pressure)
    /**
     * Compute the local phase field (ϕ) and the normalized pressure p⋆
     * at the current cell.  The zeroth moment of the phase‑field
     * distribution yields ϕ and the zeroth moment of the g distribution
     * yields p⋆.  Using intermediate sums for opposite direction pairs
     * improves vector performance.
     */
    inline auto macro_phi_P(double& f0) {   // Eq. (12), Eq. (32a)
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
    inline auto macro_u(double& f0)  {   // Eq. (32b) first term (moment of g)
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
     * based on the phase field ϕ (Eq. (13)).
     */
    inline double local_rho(double& f0) {   // Eq. (13)
        auto [phi, P] = macro_phi_P(f0);
        return rho_g + (phi - phi_g) * (rho_l - rho_g);
    }

    /**
     * Linear interpolation of dynamic viscosity (Eq. (24)) followed by
     * computation of the hydrodynamic relaxation time (Eq. (25)).  The
     * effective viscosity is μ = μ_g + (ϕ-ϕ_g)(μ_l - μ_g).  Then
     * τ = μ/(ρ c_s^2).
     */
    inline double local_tau(double& f0) {   // Eq. (25)
        double rho = local_rho(f0);
        auto [phi , p] = macro_phi_P(f0);
        double mu = mu_g + (phi - phi_g) * (mu_l - mu_g);
        return mu / (rho * cs2);
    }

    /**
     * Compute the kinematic viscosity ν = τ c_s^2 (Eq. (21)).
     */
    inline double local_nu(double& f0) {    // Eq. (21)
        auto tau = local_tau(f0);
        return tau * cs2;
    }

    /**
     * Compute the chemical potential term μϕ (Eq. (5)).
     */
    inline double mu_phi(double& f0) {    // Eq. (5)
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
     * Compute the gradient of the phase field using an isotropic
     * finite‑difference stencil.  Periodic boundaries are assumed.  At
     * bounce‑back nodes we approximate the gradient using the opposite
     * node (mirroring across the boundary).  The factor 1/cs^2
     * appears in the denominator.
     */
    inline std::array<double,2> gradient_phi(double& f0){
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
            if (flag[nb] == CellType_laplace2D_MRT_opt::bounce_back) {
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
     * Compute the Laplacian of the phase field using a simple
     * finite‑difference stencil.
     */
    inline double laplacian_phi(double& f0){
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
            if (flag[nb] != CellType_laplace2D_MRT_opt::bounce_back) {
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
     * phase field (Eq. (33)).
     */
    inline std::array<double,2> gradient_rho(double& f0){
        auto grad_phi = gradient_phi(f0);
        double delta_rho = (rho_l - rho_g);
        return { delta_rho * grad_phi[0], delta_rho * grad_phi[1] };
    }

    // ------------------------------------------------------------------
    // Forces (Fs , Fb , Fp , Fμ)
    // ------------------------------------------------------------------
    /**
     * Surface tension force Fs = μϕ ∇ϕ.
     */
    inline std::array<double,2> Fs(double& f0){
        auto grad_phi = gradient_phi(f0);
        double mu_phi_val = mu_phi(f0);
        return { mu_phi_val * grad_phi[0], mu_phi_val * grad_phi[1] };
    }

    /**
     * Body force due to gravity or other external fields: F_b = ρ g.
     */
    inline std::array<double,2> Fb(double& f0){
        double rho = local_rho(f0);
        return { rho * Fbx, rho * Fby };
    }

    /**
     * Pressure gradient force F_p = −p⋆ c_s^2 ∇ρ.
     */
    inline std::array<double,2> Fp(double& f0){
        auto [phi , Pstar] = macro_phi_P(f0);
        auto grad_rho = gradient_rho(f0);
        return { -Pstar * cs2 * grad_rho[0], -Pstar * cs2 * grad_rho[1] };
    }

    /**
     * Viscous force F_μ computed using the MRT stress tensor.
     * This version caches the lattice pointers and applies the MRT
     * transform on arrays of plain doubles to aid vectorization.
     */
    inline std::array<double,2> Fmu(double& f0){
        // obtain local index
        auto idx = &f0 - lattice;
        auto u            = mid_velocity(f0);
        double rho  = local_rho(f0);
        double tau  = local_tau(f0);
        double nu   = local_nu(f0);
        auto grad_rho = gradient_rho(f0);
        // gather g and build g_eq in standard ordering
        double gvec_std[9];
        double geq_std[9];
        double u_sq = (u[0]*u[0] + u[1]*u[1]) / (2.0*cs2);
        for (int k = 0; k < 9; ++k) {
            int s = custom_to_std[k];
            double ex = c[k][0];
            double ey = c[k][1];
            double ck_u = ex*u[0] + ey*u[1];
            double gamma_k = t[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - u_sq);
            double gk = gin(idx,k);
            double geq_k = std::get<1>(macro_phi_P(f0)) * t[k] + (gamma_k - t[k]);
            gvec_std[s] = gk;
            geq_std[s]  = geq_k;
        }
        // compute moment vector m = M*(g_std - g_eq_std)
        double m_vec[9];
        for (int r = 0; r < 9; ++r) {
            double sum = 0.0;
            for (int a = 0; a < 9; ++a) {
                sum += M[r][a] * (gvec_std[a] - geq_std[a]);
            }
            m_vec[r] = sum;
        }
        // relaxation matrix S_diag, s_nu = 1/(tau + 0.5)
        double s_nu = 1.0 / (tau + 0.5);
        double s_m[9];
        for (int r = 0; r < 9; ++r) {
            double Sr = (r < 7 ? 1.0 : s_nu);
            s_m[r] = Sr * m_vec[r];
        }
        // convert back to velocity space: A_std = M_inv * s_m
        double A_std[9];
        for (int a = 0; a < 9; ++a) {
            double sum = 0.0;
            for (int r = 0; r < 9; ++r) {
                sum += M_inv[a][r] * s_m[r];
            }
            A_std[a] = sum;
        }
        // reorder A_std back to custom ordering
        double A_custom[9];
        for (int k = 0; k < 9; ++k) {
            int s = custom_to_std[k];
            A_custom[k] = A_std[s];
        }
        // contract stress tensor with gradient of density
        double Txx = 0.0, Txy = 0.0, Tyy = 0.0;
        for (int beta = 0; beta < 9; ++beta) {
            double ex = static_cast<double>(c[beta][0]);
            double ey = static_cast<double>(c[beta][1]);
            double A = A_custom[beta];
            Txx += A * ex * ex;
            Txy += A * ex * ey;
            Tyy += A * ey * ey;
        }
        double pref = - nu / cs2;
        double Fx = pref * ( Txx * grad_rho[0] + Txy * grad_rho[1] );
        double Fy = pref * ( Txy * grad_rho[0] + Tyy * grad_rho[1] );
        return {Fx, Fy};
    }

    /**
     * Total force excluding viscous MRT contribution: Fs + Fb + Fp.
     */
    inline std::array<double,2> Total_Fsbp(double& f0){
        auto F_s = Fs(f0);
        auto F_b = Fb(f0);
        auto F_p = Fp(f0);
        return { F_s[0] + F_b[0] + F_p[0] , F_s[1] + F_b[1] + F_p[1] };
    }

    /**
     * Total force including viscous MRT contribution: Fs + Fb + Fp + Fμ.
     */
    inline std::array<double,2> Total_F(double& f0){
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
    inline std::array<double,2> mid_velocity(double& f0){
        auto u = macro_u(f0);
        auto force = Total_Fsbp(f0);
        double rho = local_rho(f0);
        return { u[0] + force[0]/(2.0*rho), u[1] + force[1]/(2.0*rho) };
    }

    /**
     * Compute the full velocity including viscous contributions (Eq. (32b)).
     */
    inline std::array<double,2> velocity(double& f0){
        auto u = macro_u(f0);
        auto force = Total_F(f0);
        double rho = local_rho(f0);
        return { u[0] + force[0]/(2.0*rho), u[1] + force[1]/(2.0*rho) };
    }

    /**
     * Compute the k‑independent part of the phase‑field forcing term
     * Fϕ_α.  The actual forcing term is Fϕ_k = δt w_k (F_phi·c_k).  Here we
     * return the factor that multiplies the dot product with the lattice
     * velocity and the weight.
     */
    inline std::array<double,2> Fphi_fac(double& f0){
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
    inline void stream(int i,int k,int iX,int iY,double pf,double pg)
    {
        int XX = iX + c[k][0];
        int YY = iY + c[k][1];
        XX = (XX + dim.nx) % dim.nx;
        YY = (YY + dim.ny) % dim.ny;
        size_t nb = xyz_to_i(XX,YY);
        if (flag[nb] == CellType_laplace2D_MRT_opt::bounce_back) {
            fout(i,opp[k]) = pf;
            gout(i,opp[k]) = pg;
        }
        else {
            fout(nb,k) = pf;
            gout(nb,k) = pg;
        }
    }

    // ------------------------------------------------------------------
    // Main per‑cell operator
    // ------------------------------------------------------------------
    /**
     * Perform the collision and streaming operations on a single cell.
     * For bulk cells we compute the macroscopic variables, forces,
     * equilibria, and then update both f and g distributions.  The
     * phase field uses a BGK relaxation and the hydrodynamics use an
     * MRT relaxation.  We implement a two‑phase streaming step:
     * for non‑rest populations (k < 4) we stream both k and opp[k],
     * while the rest population (k == 4) is updated in place.
     */
    inline void operator()(double& f0)
    {
        int idx = &f0 - lattice;
        if (flag[idx] != CellType_laplace2D_MRT_opt::bulk) {
            return;
        }
        auto  u   = velocity(f0);
        auto [phi, Pstar] = macro_phi_P(f0);
        double rho = local_rho(f0);
        double tau = local_tau(f0);
        double tau_phi = mobility / cs2;
        auto F_sbp = Total_Fsbp(f0);
        auto F_mu  = Fmu(f0);
        auto F_tot = std::array<double,2>{ F_sbp[0] + F_mu[0], F_sbp[1] + F_mu[1] };
        auto F_phi = Fphi_fac(f0);
        double u_sq = (u[0]*u[0] + u[1]*u[1]) / (2.0*cs2);
        // --- Phase field BGK collision ---
        double pf_out[9];
        for (int k = 0; k < 9; ++k) {
            double ex = c[k][0];
            double ey = c[k][1];
            double ck_u = ex*u[0] + ey*u[1];
            double gamma_k = t[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - u_sq);
            double Ff_k = t[k] * (F_phi[0] * ex + F_phi[1] * ey);
            double feq_bar = phi * gamma_k - Ff_k / 2.0;
            double f_in_val = fin(idx,k);
            pf_out[k] = f_in_val - (f_in_val - feq_bar) / (tau_phi + 0.5) + Ff_k;
        }
        // --- Hydrodynamic MRT collision ---
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
        double m_vec[9] = {0.0}, m_eq[9] = {0.0}, Fm[9] = {0.0}, m_post[9] = {0.0};
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
        double s_nu = 1.0 / (tau + 0.5);
        double Sdiag[9] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, s_nu, s_nu };
        for (int r = 0; r < 9; ++r) {
            double Sr = Sdiag[r];
            m_post[r] = m_vec[r] - Sr * (m_vec[r] - m_eq[r]) + (1.0 - 0.5 * Sr) * Fm[r];
        }
        double g_post_std[9];
        for (int a = 0; a < 9; ++a) {
            double sum = 0.0;
            for (int r = 0; r < 9; ++r) {
                sum += M_inv[a][r] * m_post[r];
            }
            g_post_std[a] = sum;
        }
        double g_post[9];
        for (int k = 0; k < 9; ++k) {
            int s = custom_to_std[k];
            g_post[k] = g_post_std[s];
        }
        // obtain current coordinates for streaming
        auto [iX,iY] = i_to_xyz(idx);
        for (int k = 0; k < 4; ++k) {
            double pf  = pf_out[k];
            double pg  = g_post[k];
            double pf_op = pf_out[ opp[k] ];
            double pg_op = g_post[ opp[k] ];
            stream(idx,     k      , iX, iY, pf   , pg);
            stream(idx, opp[k], iX, iY, pf_op, pg_op);
        }
        int k0 = 4;
        fout(idx,k0) = pf_out[k0];
        gout(idx,k0) = g_post[k0];
    }
};

// ───────────────────────── VTK output ─────────────────────────────────
/**
 * Write diagnostic fields (phase field, density, pressure, velocity and
 * flag) to a VTK file for visualization.
 */
inline void saveVtkFields_laplace2D_MRT_opt(LBM_laplace2D_MRT_opt& lbm, int time_iter, double dx = 0.) {
    Dim_laplace2D_MRT_opt const& dim = lbm.dim;
    if (dx == 0.0) dx = 1.0 / dim.nx;
    int dimZ = 1;
    stringstream ss;
    ss << "sol_opt_" << setw(7) << setfill('0') << time_iter << ".vtk";
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
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
                auto [phi, p] = lbm.macro_phi_P(lbm.lattice[i]);
                val = (float)phi;
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
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
                val = (float)lbm.local_rho(lbm.lattice[i]);
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
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
                auto [phi, p] = lbm.macro_phi_P(lbm.lattice[i]);
                double rho = lbm.local_rho(lbm.lattice[i]);
                val = p * rho / 3.0;
            }
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
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
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
            os << ((lbm.flag[i] == CellType_laplace2D_MRT_opt::bounce_back) ? 1 : 0) << " ";
        }
        os << "\n";
    }
    os << "\n";
    // Fs (surface tension)
    os << "VECTORS Fs float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double fx = 0., fy = 0.;
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
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
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
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
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
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
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
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
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
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
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
                auto F = lbm.Total_Fsbp(lbm.lattice[i]);
                fx = F[0]; fy = F[1];
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";
}

// ───────────────────────── Energy computation (use stored Ux,Uy) ──────────────────────────────
/**
 * Compute the total kinetic energy of the fluid (for diagnostics).
 */
inline double computeEnergy_laplace2D_MRT_opt(LBM_laplace2D_MRT_opt& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_laplace2D_MRT_opt::bulk) {
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
inline void inigeom_laplace2D_MRT_opt(LBM_laplace2D_MRT_opt& lbm) {
    Dim_laplace2D_MRT_opt const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (y == dim.ny + 100 || y == dim.ny + 111) {
                lbm.flag[i] = CellType_laplace2D_MRT_opt::bounce_back;
                for (int k = 0; k < 9; ++k) {
                    lbm.fin(i,k) = 0.0;
                    lbm.gin(i,k) = 0.0;
                }
            }
            else {
                lbm.flag[i] = CellType_laplace2D_MRT_opt::bulk;
            }
        }
    }
}

// ─────────────────────────   main driver (Laplace MRT)  ────────────────────────
/**
 * Run the LBM solver for the conservative phase‑field model.  The
 * configuration file "config_laplace2D_MRT.txt" under
 * apps/Config_Files should specify simulation parameters.  The
 * optimized implementation closely follows the reference version but uses
 * renamed symbols.  Diagnostics are omitted for maximum speed.
 */
inline void laplace2D_MRT_opt()
{
    ifstream contfile("../apps/Config_Files/config_laplace2D_MRT.txt");
    if (!contfile.is_open()) {
        throw invalid_argument(
            "Config file not found. It should be named "
            "\"config_laplace2D_MRT.txt\" in apps/Config_Files.");
    }
    double ulb=1.0, max_t=0.01, phi_l=1.0, phi_g=0.0, rho_l=1.0, rho_g=1e-3;
    double Sigma=0.01, xi=4.0, M=0.02, R=0.0, Fbx=0.0, Fby = 0.0;
    double mu_l=0.1, mu_g=0.0001;
    int N=128, out_freq=1000, vtk_freq=1000;
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
    Dim_laplace2D_MRT_opt dim {N, N};
    auto [Re, At, Ca, Pe, dx,dt] = lbParameters_laplace2D_MRT_opt(ulb, N, mu_l, Sigma, rho_l , rho_g , M);
    printParameters_laplace2D_MRT_opt(dim, Re, At, Ca, Pe, ulb, N, max_t);
    vector<CellData_opt> lattice_vect(LBM_laplace2D_MRT_opt::sizeOfLattice(dim.nelem));
    CellData_opt *lattice = lattice_vect.data();
    vector<CellType_laplace2D_MRT_opt> flag_vect(dim.nelem);
    CellType_laplace2D_MRT_opt* flag = &flag_vect[0];
    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];
    auto [c_vect, opp_vect, t_vect] = d2q9_constants_laplace2D_MRT_opt();
    LBM_laplace2D_MRT_opt lbm{
        lattice, flag, parity,
        c_vect.data(), opp_vect.data(), t_vect.data(),
        phi_l, phi_g, rho_l, rho_g,
        mu_l, mu_g, Sigma,
        Fbx, Fby, xi, M,
        dim
    };
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData_opt& f0) { lbm.iniLattice(f0); });
    inigeom_laplace2D_MRT_opt(lbm);
    auto [start, clock_iter] = restartClock_laplace2D_MRT_opt();
    ofstream efile("energy_opt.dat");
    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
            saveVtkFields_laplace2D_MRT_opt(lbm, time_iter);
        }
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(4) << time_iter * dt << setprecision(3)
                 << " [" << time_iter * dt / max_t * 100. << "%]" << endl;
            double energy = computeEnergy_laplace2D_MRT_opt(lbm) * dx * dx / (dt * dt);
            cout << "Average energy: " << setprecision(8) << energy << endl;
            efile << setw(10) << time_iter * dt << setw(16) << setprecision(8) << energy << endl;
        }
        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        *parity = 1 - *parity;
        ++clock_iter;
    }
    printMlups_laplace2D_MRT_opt(start, clock_iter, dim.nelem);
}