#pragma once
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>
#include <functional>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>

using namespace std;
using namespace std::chrono;

// ============================================================================
// Rayleigh-Taylor instability (RTI) in 2D, using the Liang et al. (2018)
// conservative Allen-Cahn / BGK-BGK phase-field LB model (PRE 97, 033309).
// Class/function names follow the "..._rayleighTaylor2D_Fakhari" convention
// requested by the user (naming scheme reused from a companion Fakhari-style
// app), but the physics (Eqs. 1-28 referenced below) is the Liang model,
// carried over — with all Laplace-law fixes — from laplace2D_Liang.h.
// ============================================================================

enum class CellType_rayleighTaylor2D_Fakhari : uint8_t { bounce_back, bulk };
using CellData = double;

inline auto d2q9_constants_rayleighTaylor2D_Fakhari() {
    vector<array<int,2>> c_vect = {
        {-1, 0}, { 0,-1}, {-1,-1}, {-1, 1}, { 0, 0},
        { 1, 0}, { 0, 1}, { 1, 1}, { 1,-1}
    };
    vector<int> opp_vect = { 5, 6, 7, 8, 4, 0, 1, 2, 3 };
    vector<double> t_vect = {
        1./9., 1./9., 1./36., 1./36., 4./9.,
        1./9., 1./9., 1./36., 1./36.
    };
    return make_tuple(c_vect, opp_vect, t_vect);
}

struct Dim_rayleighTaylor2D_Fakhari {
    Dim_rayleighTaylor2D_Fakhari(int nx_, int ny_)
    : nx(nx_), ny(ny_), nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)), npop(9 * nelem) {}
    int nx, ny;
    size_t nelem, npop;
};

// Diagnostic-only nondimensional groups (do NOT feed back into the solver,
// which always runs in lattice units with dx = dt = 1, cs2 = 1/3).
// g_lat        : gravitational acceleration magnitude, in lattice units
// lref         : reference length (domain width, nx)
// nu_l         : reference (heavy-phase) kinematic viscosity, lattice units
inline auto lbParameters_rayleighTaylor2D_Fakhari(double g_lat, int lref, double nu_l, double Sigma,
                                                   double rho_l, double rho_g, double M) {
    double Uref = sqrt(g_lat * static_cast<double>(lref));
    double Re = pow(static_cast<double>(lref), 1.5) * sqrt(g_lat) / nu_l;
    double At = (rho_l - rho_g) / (rho_l + rho_g);
    double Ca = nu_l * rho_l * Uref / Sigma;
    double Pe = Uref * static_cast<double>(lref) / M;
    double dx = 1. / static_cast<double>(lref);
    double dt = dx * Uref;
    return make_tuple(Re, At, Ca, Pe, Uref, dx, dt);
}

inline void printParameters_rayleighTaylor2D_Fakhari(const Dim_rayleighTaylor2D_Fakhari &dim,
                                                       double Re, double At, double Ca, double Pe,
                                                       double Uref, double g_lat, double max_t) {
    cout << "Rayleigh-Taylor 2-D (Liang conservative Allen-Cahn -- BGK/BGK)\n"
         << "nx     = " << dim.nx << '\n'
         << "ny     = " << dim.ny << '\n'
         << "Re     = " << Re << '\n'
         << "At     = " << At << '\n'
         << "Ca     = " << Ca << '\n'
         << "Pe     = " << Pe << '\n'
         << "Uref   = " << Uref << '\n'
         << "g      = " << g_lat << '\n'
         << "max_t  = " << max_t << '\n';
}

inline auto restartClock_rayleighTaylor2D_Fakhari() {
    return make_pair(high_resolution_clock::now(), 0);
}

template <typename TP>
inline void printMlups_rayleighTaylor2D_Fakhari(TP start, int iter, size_t nelem) {
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * static_cast<size_t>(iter)) / us.count();
    cout << "Runtime: " << us.count()/1000000.0 << " seconds " << endl;
    cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

struct LBM_rayleighTaylor2D_Fakhari {
    using CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem + 2 * 9 * nelem; }

    CellData* lattice;
    CellType_rayleighTaylor2D_Fakhari* flag;
    int* parity;
    array<int,2>* c;
    int* opp;
    double* t;
    double* phiu_prev_x;
    double* phiu_prev_y;
    int* nb;
    int* nbb;
    double phi_l, phi_g, rho_l, rho_g, nu_l, nu_g, Sigma, Fbx, Fby, xi, M;
    Dim_rayleighTaylor2D_Fakhari dim;

    static constexpr double cs2 = 1.0/3.0;
    static constexpr double PI = 3.1415926535897932384626433832795;

    inline size_t xyz_to_i(int x, int y) const { return static_cast<size_t>(y + dim.ny * x); }
    inline auto i_to_xyz(int i) const { return make_tuple(i / dim.ny, i % dim.ny); }

    inline double& fin (int i,int k) { return lattice[(*parity)*dim.npop + k*dim.nelem + i]; }
    inline double& fout(int i,int k) { return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }

    inline double& gin (int i,int k) { return lattice[(*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    inline double& gout(int i,int k) { return lattice[(1-*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    inline double fin (int i,int k) const { return lattice[(*parity)*dim.npop + k*dim.nelem + i]; }
    inline double gin (int i,int k) const { return lattice[(*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    // ───────────────────────── Neighbor tables (streaming only) ─────────────
    // Periodic in BOTH x and y here; the y-periodicity is harmless because
    // walls are flagged bounce_back exactly at y=0 and y=ny-1 (domain edges,
    // no ghost padding), and bounce-back nodes are never streamed FROM
    // (operator() early-returns for non-bulk cells). Interior bulk cells
    // adjacent to a wall (y=1 or y=ny-2) never trigger the modulo wraparound.
    inline void build_neighbors() {
        for (int x = 0; x < dim.nx; ++x) {
            for (int y = 0; y < dim.ny; ++y) {
                int i = static_cast<int>(xyz_to_i(x,y));
                for (int k=0; k<9; ++k) {
                    int XX  = (x + c[k][0] + dim.nx) % dim.nx;
                    int YY  = (y + c[k][1] + dim.ny) % dim.ny;
                    int XXb = (x - c[k][0] + dim.nx) % dim.nx;
                    int YYb = (y - c[k][1] + dim.ny) % dim.ny;
                    nb [i*9 + k] = static_cast<int>(xyz_to_i(XX ,YY ));
                    nbb[i*9 + k] = static_cast<int>(xyz_to_i(XXb,YYb));
                }
            }
        }
    }

    // ───────────────────────── Initial condition: perturbed RTI interface ───
    // Heavy fluid (phi -> phi_l) on top, light fluid (phi -> phi_g) on bottom,
    // single-mode cosine perturbation of wavelength = domain width (Lx = nx).
    // Matches the classic single-mode RTI benchmark setup.
    inline void iniLattice(double& f0) {
        int i = static_cast<int>(&f0 - lattice);
        auto [iX, iY] = i_to_xyz(i);

        double phi0 = 0.5*(phi_l + phi_g);
        double amp  = 0.5*(phi_l - phi_g);
        double Ly = static_cast<double>(dim.ny);
        double Lx = static_cast<double>(dim.nx);

        double amplitude = 0.1 * Lx;
        double xcell = double(iX) + 0.5;
        double ycell = double(iY) + 0.5;
        double x_norm = xcell / Lx;
        double y_interface = 0.5*(Ly - 1.0) + amplitude * std::cos(2.0*PI*x_norm);

        double phi = phi0 + amp * std::tanh((ycell - y_interface) / (0.5*xi));

        for (int k = 0; k < 9; ++k) {
            fin(i,k) = phi * t[k];
            gin(i,k) = 0.0;
        }
    }

    inline double macro_phi(int i) const { // Eq. (12): phi = sum_i f_i
        double phi = 0.0; for (int k=0; k<9; ++k) phi += fin(i,k); return phi;
    }

    // Eq. (13): rho = phi*(rho_l - rho_g) + rho_g
    inline double rho_from_phi(double phi) const { return rho_g + phi * (rho_l - rho_g); }

    // Eq. (24): nu = phi*(nu_l - nu_g) + nu_g
    inline double nu_from_phi(double phi) const { return nu_g + phi * (nu_l - nu_g); }

    // Eq. (23): nu = cs^2 (tau_g - 0.5) dt  =>  tau_g = nu/cs^2 + 0.5
    inline double tau_g_from_phi(double phi) const { return nu_from_phi(phi) / cs2 + 0.5; }

    // Eq. (14): M = cs^2 (tau_f - 0.5) dt   =>  tau_f = M/cs^2 + 0.5
    inline double tau_f_from_M() const { return M / cs2 + 0.5; }

    // Eq. (3): lambda = 4 phi (1-phi) / xi
    inline double lambda_phi(double phi) const { return 4.0 * phi * (1.0 - phi) / xi; }

    // Body force per unit volume = rho * g_vec  (Fbx, Fby are gravitational
    // ACCELERATION components; must be weighted by local density to give a
    // correct buoyancy-driven RTI -- a constant, density-independent force
    // would make both phases feel identical forcing and suppress the
    // instability's density-contrast-driven growth mechanism).
    inline array<double,2> body_force(double rho) const { return {rho*Fbx, rho*Fby}; }

    inline array<double,2> unit_normal_from_gradphi(const array<double,2>& grad_phi) const {
        double s = sqrt(grad_phi[0]*grad_phi[0] + grad_phi[1]*grad_phi[1]);
        if (s < 1e-30) return {0.0, 0.0};
        return {grad_phi[0]/s, grad_phi[1]/s};
    }

    inline array<double,2> dphiu_dt(int i, double phi, const array<double,2>& u) const {
        return {(phi*u[0] - phiu_prev_x[i]), (phi*u[1] - phiu_prev_y[i])};
    }

    inline void store_phiu_prev(int i, double phi, const array<double,2>& u) {
        phiu_prev_x[i] = phi * u[0];
        phiu_prev_y[i] = phi * u[1];
    }

    // ───────────────────────── Differential operators ───────────────────────
    // Both grad_phi and laplacian_phi already implement a no-flux / mirror
    // treatment of phi at any bounce_back neighbor (including out-of-range
    // y, which cannot occur here since walls occupy y=0 and y=ny-1 exactly).
    // This makes them directly usable at the RTI channel walls with no
    // modification.
    inline array<double,2> grad_phi(int i) const {
        auto [iX, iY] = i_to_xyz(i);
        array<double,2> gphi{0.0, 0.0};

        for (int k = 0; k < 9; ++k) {
            if (k == 4) continue;

            int ix = (iX + c[k][0] + dim.nx) % dim.nx;
            int iy = iY + c[k][1];

            double phi_nb = 0.0;

            if (iy < 0 || iy >= dim.ny) {
                int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                int iybb = iY - c[k][1];
                int nbb_ = (int)xyz_to_i(ixbb, iybb);
                phi_nb = macro_phi(nbb_);
            } else {
                int nb_ = (int)xyz_to_i(ix, iy);
                if (flag[nb_] == CellType_rayleighTaylor2D_Fakhari::bounce_back) {
                    int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                    int iybb = iY - c[k][1];
                    int nbb_ = (int)xyz_to_i(ixbb, iybb);
                    phi_nb = macro_phi(nbb_);
                } else {
                    phi_nb = macro_phi(nb_);
                }
            }
            gphi[0] += t[k] * c[k][0] * phi_nb;
            gphi[1] += t[k] * c[k][1] * phi_nb;
        }
        gphi[0] /= cs2;
        gphi[1] /= cs2;
        return gphi;
    }

    // Eq. (28)
    inline double laplacian_phi(int i) const {
        auto [iX, iY] = i_to_xyz(i);
        double phi_c = macro_phi(i);
        double lap = 0.0;

        for (int k = 0; k < 9; ++k) {
            if (k == 4) continue;

            int ix = (iX + c[k][0] + dim.nx) % dim.nx;
            int iy = iY + c[k][1];

            double phi_nb = 0.0;

            if (iy < 0 || iy >= dim.ny) {
                int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                int iybb = iY - c[k][1];
                int nbb_ = (int)xyz_to_i(ixbb, iybb);
                phi_nb = macro_phi(nbb_);
            } else {
                int nb_ = (int)xyz_to_i(ix, iy);
                if (flag[nb_] == CellType_rayleighTaylor2D_Fakhari::bounce_back) {
                    int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                    int iybb = iY - c[k][1];
                    int nbb_ = (int)xyz_to_i(ixbb, iybb);
                    phi_nb = macro_phi(nbb_);
                } else {
                    phi_nb = macro_phi(nb_);
                }
            }
            lap += 2.0 * t[k] * (phi_nb - phi_c);
        }
        lap /= cs2;
        return lap;
    }

    inline array<double,2> gradient_rho_from_gradphi(const array<double,2>& grad_phi) const {
        double delta_rho = rho_l - rho_g;
        return { delta_rho * grad_phi[0], delta_rho * grad_phi[1] };
    }

    inline array<double,2> normal_phi(int i) const {
        auto gphi = grad_phi(i);
        double mag = std::sqrt(gphi[0]*gphi[0] + gphi[1]*gphi[1]);
        if (mag < 1e-14) return {0.0, 0.0};
        return {gphi[0]/mag, gphi[1]/mag};
    }

    // Eq. (6)-(7): mu_phi = 4 beta phi(phi-1)(phi-0.5) - k laplacian(phi)
    inline double chemical_potential(int i) const {
        double phi = macro_phi(i);
        double lap = laplacian_phi(i);
        double kappa = 1.5 * Sigma * xi;
        double beta  = 12.0 * Sigma / xi;
        return 4.0 * beta * phi * (phi - 1.0) * (phi - 0.5) - kappa * lap;
    }

    inline array<double,2> surface_force(int i) const {
        auto gphi = grad_phi(i);
        double mu_phi = chemical_potential(i);
        return {mu_phi * gphi[0], mu_phi * gphi[1]};
    }

    // Eq. (19): F = Fs + G, with G = rho * g_vec (buoyancy-correct body force)
    inline array<double,2> F_total_liang(double mu_phi_val, const array<double,2>& grad_phi, double rho) const {
        array<double,2> Fs{mu_phi_val * grad_phi[0], mu_phi_val * grad_phi[1]};
        auto Gb = body_force(rho);
        return {Fs[0] + Gb[0], Fs[1] + Gb[1]};
    }

    // Eq. (22a): rho*u = sum_i c_i g_i + 0.5*dt*F
    inline array<double,2> macro_u(int i) const {
        double phi = macro_phi(i);
        double rho = rho_from_phi(phi);
        array<double,2> mom{0.0, 0.0};
        for (int k=0; k<9; ++k) {
            mom[0] += c[k][0] * gin(i,k);
            mom[1] += c[k][1] * gin(i,k);
        }
        auto gradPhi = grad_phi(i);
        double mu_phi = chemical_potential(i);
        auto Ftot = F_total_liang(mu_phi, gradPhi, rho);
        mom[0] += 0.5 * Ftot[0];
        mom[1] += 0.5 * Ftot[1];
        return {mom[0] / rho, mom[1] / rho};
    }

    // Eq. (21b)/(22b): hydrodynamic pressure
    inline double pressure_liang(int i, double rho, const array<double,2>& u, const array<double,2>& grad_rho) const {
        double sum_nonzero = 0.0;
        for (int k = 0; k < 9; ++k) if (k != 4) sum_nonzero += gin(i,k);
        double s0 = t[4] * ( - (u[0]*u[0] + u[1]*u[1]) / (2.0*cs2) );
        return cs2 / (1.0 - t[4]) * (sum_nonzero + 0.5 * (u[0]*grad_rho[0] + u[1]*grad_rho[1]) + rho * s0);
    }

    // Convenience accessor used by the interface tracker: returns {phi, P}.
    inline pair<double,double> macro_phi_P(int i) const {
        double phi = macro_phi(i);
        if (flag[i] != CellType_rayleighTaylor2D_Fakhari::bulk) return {phi, 0.0};
        double rho = rho_from_phi(phi);
        auto gradPhi = grad_phi(i);
        auto grad_rho = gradient_rho_from_gradphi(gradPhi);
        auto u = macro_u(i);
        double p = pressure_liang(i, rho, u, grad_rho);
        return {phi, p};
    }

    // Eq. (11)
    inline array<double,2> liang_phase_source_vector(int i, double phi, const array<double,2>& grad_phi, const array<double,2>& u) const {
        auto n = unit_normal_from_gradphi(grad_phi);
        double lam = lambda_phi(phi);
        auto ddt = dphiu_dt(i, phi, u);
        return {ddt[0] + cs2 * lam * n[0], ddt[1] + cs2 * lam * n[1]};
    }

    // Eq. (8)
    inline double Fi_liang(int k, double tau_f, const array<double,2>& src_vec) const {
        double pref = (1.0 - 0.5 / tau_f) * t[k] / cs2;
        return pref * (c[k][0] * src_vec[0] + c[k][1] * src_vec[1]);
    }

    // Eq. (17)
    inline double si_liang(int k, const array<double,2>& u) const {
        double ck_u = c[k][0]*u[0] + c[k][1]*u[1], uu = u[0]*u[0] + u[1]*u[1];
        return t[k] * ( ck_u / cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - uu/(2.0*cs2) );
    }

    // Eq. (18) — FIXED: uses the total force F = Fs + G (surface tension +
    // buoyancy-correct gravity), and the tensor term matches Eq. (18) exactly
    // (no spurious extra u.grad(rho) term).
    inline double Gi_liang(int k, double tau_g,
                            const array<double,2>& u,
                            const array<double,2>& Ftot,
                            const array<double,2>& grad_rho) const {
        double ci_dot_F = c[k][0]*Ftot[0] + c[k][1]*Ftot[1];

        double tensor =
              (u[0]*grad_rho[0]) * (c[k][0]*c[k][0] - cs2)
            + (u[0]*grad_rho[1]) * (c[k][0]*c[k][1])
            + (u[1]*grad_rho[0]) * (c[k][1]*c[k][0])
            + (u[1]*grad_rho[1]) * (c[k][1]*c[k][1] - cs2);

        return (1.0 - 0.5 / tau_g) * t[k] * (ci_dot_F / cs2 + tensor / cs2);
    }

    inline void stream_tab(int i,int k,double pf,double pg) {
        const int j = nb[i*9 + k];
        if (flag[j] == CellType_rayleighTaylor2D_Fakhari::bounce_back) {
            fout(i, opp[k]) = pf;
            gout(i, opp[k]) = pg;
        } else {
            fout(j, k) = pf;
            gout(j, k) = pg;
        }
    }

    // Eq. (9), Eq. (16)
    inline tuple<double,double> collideBgkLiang(int i,int k, const array<double,2>& u,double rho,
                                                 double phi, double p, double tau_g, double tau_f,
                                                 const array<double,2>& Ftot, const array<double,2>& phase_src,
                                                 const array<double,2>& grad_rho) const {
        double feq = t[k] * phi * (1.0 + (c[k][0]*u[0] + c[k][1]*u[1]) / cs2);
        double geq = (k == 4) ? (p / cs2) * (t[k] - 1.0) + rho * si_liang(k, u)
                              : (p / cs2) * t[k] + rho * si_liang(k, u);
        double Fi = Fi_liang(k, tau_f, phase_src);
        double Gi = Gi_liang(k, tau_g, u, Ftot, grad_rho);
        double pf = fin(i,k) - (fin(i,k) - feq) / tau_f + Fi;
        double pg = gin(i,k) - (gin(i,k) - geq) / tau_g + Gi;
        return {pf, pg};
    }

    inline void operator()(double& f0) {
        int i = static_cast<int>(&f0 - lattice);
        if (flag[i] != CellType_rayleighTaylor2D_Fakhari::bulk) return;
        double phi = macro_phi(i);
        double rho = rho_from_phi(phi);
        double tau_g = tau_g_from_phi(phi);
        double tau_f_loc = tau_f_from_M();
        auto gradPhi = grad_phi(i);
        auto grad_rho = gradient_rho_from_gradphi(gradPhi);
        double mu_phi_val = chemical_potential(i);
        auto Ftot = F_total_liang(mu_phi_val, gradPhi, rho);
        auto u = macro_u(i);
        double p = pressure_liang(i, rho, u, grad_rho);
        auto phase_src = liang_phase_source_vector(i, phi, gradPhi, u);
        for (int k = 0; k < 9; ++k) {
            auto [pf, pg] = collideBgkLiang(i, k, u, rho, phi, p, tau_g, tau_f_loc, Ftot, phase_src, grad_rho);
            stream_tab(i, k, pf, pg);
        }
        store_phiu_prev(i, phi, u);
    }
};

// ───────────────────────── Geometry initialization ─────────────────────────
// Periodic in x; solid (bounce-back) walls at y = 0 and y = dim.ny - 1.
inline void inigeom_rayleighTaylor2D_Fakhari(LBM_rayleighTaylor2D_Fakhari& lbm)
{
    Dim_rayleighTaylor2D_Fakhari const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (y == 0 || y == dim.ny - 1) {
                lbm.flag[i] = CellType_rayleighTaylor2D_Fakhari::bounce_back;
                for (int k = 0; k < 9; ++k) {
                    lbm.fin(i,k) = 0.0;
                    lbm.gin(i,k) = 0.0;
                }
            } else {
                lbm.flag[i] = CellType_rayleighTaylor2D_Fakhari::bulk;
            }
        }
    }
}

// ───────────────────────── Interface front tracking (RTI spike & bubble) ────
// x = 0      -> "bubble" column (light fluid rising into heavy layer, Fig. 6a)
// x = nx/2   -> "spike"  column (heavy fluid falling into light layer, Fig. 6b)
inline void findInterfaceHeights_rayleighTaylor2D_Fakhari(
    LBM_rayleighTaylor2D_Fakhari& lbm,
    double phi_l_ref,
    double phi_g_ref,
    int&    spike_y,     // output: y-index of interface at x = nx/2, -1 if none
    int&    bubble_y     // output: y-index of interface at x = 0,    -1 if none
) {
    double phi_mid = 0.5 * (phi_l_ref + phi_g_ref);
    int x_bubble = 0;
    int x_spike  = lbm.dim.nx / 2;

    spike_y  = -1;
    bubble_y = -1;

    for (int y = lbm.dim.ny - 2; y >= 1; --y) {
        int i = (int)lbm.xyz_to_i(x_bubble, y);
        auto [phi_val, P_val] = lbm.macro_phi_P(i);
        if (phi_val <= phi_mid) { bubble_y = y; break; }
    }
    for (int y = lbm.dim.ny - 2; y >= 1; --y) {
        int i = (int)lbm.xyz_to_i(x_spike, y);
        auto [phi_val, P_val] = lbm.macro_phi_P(i);
        if (phi_val <= phi_mid) { spike_y = y; break; }
    }
}

inline void saveVtkFields_rayleighTaylor2D_Fakhari(LBM_rayleighTaylor2D_Fakhari& lbm, int time_iter, double dx = 0.0) {
    Dim_rayleighTaylor2D_Fakhari const& dim = lbm.dim;
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

    os << "SCALARS phi float 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) { for (int x = 0; x < dim.nx; ++x) { int i = (int)lbm.xyz_to_i(x, y); float val = 0.f; if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) val = (float)lbm.macro_phi(i); os << val << " "; } os << "\n"; }

    os << "\nSCALARS density float 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) { for (int x = 0; x < dim.nx; ++x) { int i = (int)lbm.xyz_to_i(x, y); float val = 0.f; if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) val = (float)lbm.rho_from_phi(lbm.macro_phi(i)); os << val << " "; } os << "\n"; }

    os << "\nSCALARS pressure float 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double val = 0.0;
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                auto [phi_val, p_val] = lbm.macro_phi_P(i);
                val = p_val;
            }
            os << val << " ";
        }
        os << "\n";
    }
        os << "\nVECTORS surface_tension_force float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double fx = 0.0, fy = 0.0;
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                auto Fs = lbm.surface_force(i);   // mu_phi * grad(phi), Eq. (5)
                fx = Fs[0]; fy = Fs[1];
            }
            os << (float)fx << " " << (float)fy << " 0\n";
        }
    }

    os << "\nVECTORS body_force float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double gx = 0.0, gy = 0.0;
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                double rho = lbm.rho_from_phi(lbm.macro_phi(i));
                auto Gb = lbm.body_force(rho);    // rho * (Fbx, Fby)
                gx = Gb[0]; gy = Gb[1];
            }
            os << (float)gx << " " << (float)gy << " 0\n";
        }
    }

    os << "\nVECTORS total_force float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double Fx = 0.0, Fy = 0.0;
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                double phi = lbm.macro_phi(i);
                double rho = lbm.rho_from_phi(phi);
                auto gradPhi = lbm.grad_phi(i);
                auto Ftot = lbm.F_total_liang(lbm.chemical_potential(i), gradPhi, rho);
                Fx = Ftot[0]; Fy = Ftot[1];
            }
            os << (float)Fx << " " << (float)Fy << " 0\n";
        }
    }
    

    os << "\nVECTORS velocity float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double ux = 0.0, uy = 0.0;
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                auto u = lbm.macro_u(i);
                ux = u[0]; uy = u[1];
            }
            os << (float)ux << " " << (float)uy << " 0\n";
        }
    }

    os << "\nSCALARS Flag int 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) { for (int x = 0; x < dim.nx; ++x) { size_t i = lbm.xyz_to_i(x, y); os << ((lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bounce_back) ? 1 : 0) << " "; } os << "\n"; }
}

inline double computeEnergy_rayleighTaylor2D_Fakhari(LBM_rayleighTaylor2D_Fakhari& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                auto u = lbm.macro_u(i);
                energy += u[0]*u[0] + u[1]*u[1];
            }
        }
    }
    return 0.5 * energy;
}

inline void rayleighTaylor2D_Fakhari() {
    ifstream contfile("../apps/Config_Files/config_RTI2D_MRT.txt");
    if (!contfile.is_open()) throw invalid_argument("Config file not found: config_rayleighTaylor2D_Fakhari.txt");

    double max_t=1.0, phi_l=1.0, phi_g=0.0, rho_l=3.0, rho_g=1.0;
    double Sigma=1e-4, xi=4.0, M=0.02;
    double nu_l=0.02, nu_g=0.02/3.;
    double g_lat=1.25e-5;                 // gravitational acceleration, lattice units
    int N=128, out_freq=1000, vtk_freq=1000;

    string line, param, value;
    while (getline(contfile, line)) {
        if (auto pos = line.find('#'); pos != string::npos) line.erase(pos);
        if (line.empty()) continue;
        istringstream ls(line);
        if (!(ls >> param >> value)) continue;
        if      (param == "N")        N = stoi(value);
        else if (param == "max_t")    max_t = stod(value);
        else if (param == "out_freq") out_freq = stoi(value);
        else if (param == "vtk_freq") vtk_freq = stoi(value);
        else if (param == "phi_l")    phi_l = stod(value);
        else if (param == "phi_g")    phi_g = stod(value);
        else if (param == "rho_l")    rho_l = stod(value);
        else if (param == "rho_g")    rho_g = stod(value);
        else if (param == "Sigma")    Sigma = stod(value);
        else if (param == "xi")       xi = stod(value);
        else if (param == "M")        M = stod(value);
        else if (param == "nu_l")     nu_l = stod(value);
        else if (param == "nu_g")     nu_g = stod(value);
        else if (param == "Fy")        g_lat = stod(value);
    }

    double Fbx = 0.0, Fby = -g_lat;   // gravity points downward (-y)

    Dim_rayleighTaylor2D_Fakhari dim { N, 4*N + 2 };
    auto [Re, At, Ca, Pe, Uref, dx, dt] = lbParameters_rayleighTaylor2D_Fakhari(g_lat, N, nu_l, Sigma, rho_l, rho_g, M);
    printParameters_rayleighTaylor2D_Fakhari(dim, Re, At, Ca, Pe, Uref, g_lat, max_t);

    vector<double> lattice_vect(LBM_rayleighTaylor2D_Fakhari::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();
    vector<CellType_rayleighTaylor2D_Fakhari> flag_vect(dim.nelem);
    CellType_rayleighTaylor2D_Fakhari* flag = flag_vect.data();
    vector<int> parity_vect {0};
    int* parity = parity_vect.data();
    vector<int> nb_vect(dim.nelem*9), nbb_vect(dim.nelem*9);
    vector<double> phiu_prev_x_vect(dim.nelem, 0.0), phiu_prev_y_vect(dim.nelem, 0.0);

    auto [c_vect, opp_vect, t_vect] = d2q9_constants_rayleighTaylor2D_Fakhari();
    LBM_rayleighTaylor2D_Fakhari lbm{
        lattice, flag, parity, c_vect.data(), opp_vect.data(), t_vect.data(),
        phiu_prev_x_vect.data(), phiu_prev_y_vect.data(), nb_vect.data(), nbb_vect.data(),
        phi_l, phi_g, rho_l, rho_g, nu_l, nu_g, Sigma, Fbx, Fby, xi, M, dim
    };
    lbm.build_neighbors();
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });
    inigeom_rayleighTaylor2D_Fakhari(lbm);

    auto [start, clock_iter] = restartClock_rayleighTaylor2D_Fakhari();
    ofstream efile("energy.dat");

    ofstream ifile("interface_heights.dat");
    ifile << "# t*  spike_y  bubble_y  spike_pos/L  bubble_pos/L\n";

    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) saveVtkFields_rayleighTaylor2D_Fakhari(lbm, time_iter);
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter << ", t = " << setprecision(6) << time_iter * dt << setprecision(3)
                 << " [" << (time_iter * dt / max_t * 100.) << "%]" << endl;
            double energy = computeEnergy_rayleighTaylor2D_Fakhari(lbm) * dx * dx / (dt * dt);
            cout << "Average energy: " << setprecision(8) << energy << endl;
            efile << setw(12) << setprecision(6) << time_iter * dt << setw(18) << setprecision(8) << energy << "\n";

            int spike_y = -1, bubble_y = -1;
            findInterfaceHeights_rayleighTaylor2D_Fakhari(lbm, phi_l, phi_g, spike_y, bubble_y);

            double y_mid = 0.5 * (dim.ny - 1);
            double spike_pos  = (spike_y  >= 0) ? (spike_y  - y_mid) * dx : std::numeric_limits<double>::quiet_NaN();
            double bubble_pos = (bubble_y >= 0) ? (bubble_y - y_mid) * dx : std::numeric_limits<double>::quiet_NaN();

            ifile << setw(12) << setprecision(6) << time_iter * dt
                  << setw(8)  << spike_y
                  << setw(8)  << bubble_y
                  << setw(14) << setprecision(6) << spike_pos
                  << setw(14) << setprecision(6) << bubble_pos
                  << "\n";
        }
        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        *parity = 1 - *parity;
        ++clock_iter;
    }
    printMlups_rayleighTaylor2D_Fakhari(start, clock_iter, dim.nelem);
}
