// Modified and optimized header for porous 2D MRT LBM simulation.
// Based on "A phase‑field lattice Boltzmann model for simulating multiphase flows in porous media"
// by Fakhari et al. (2018).  This implementation uses the multiple-relaxation-time (MRT) collision
// operator as described in the paper【676137786399381†L562-L567】 and computes phase‑field gradients
// using second‑order, isotropic differences for stability【676137786399381†L626-L633】.

#pragma once

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
#include <limits>
#include <random>

// oneTBB
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/partitioner.h>

using namespace std;
using namespace std::chrono;

// ───────────────────────── type declarations ─────────────────────────
// Enumeration of cell types for porous geometry.
enum class CellType_Porous2D_MRT : uint8_t { bounce_back = 0, bulk = 1, injection = 2, outlet = 3 };
using  CellData = double;

// 2D lattice dimension helper.
struct Dim_Porous2D_MRT {
    Dim_Porous2D_MRT(int nx_, int ny_)
    : nx(nx_), ny(ny_),
      nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)),
      npop (9 * nelem) {}
    int    nx, ny;
    size_t nelem, npop;
};

// ───────────────────────── D2Q9 constants ───────────────────────────
// Return the discrete velocity directions, their opposite indices and weights.
inline auto d2q9_constants_Porous2D_MRT()
{
    // k=4 is the rest (center).
    vector<array<int,2>> c_vect = {
        {-1,  0}, { 0, -1}, {-1, -1}, {-1,  1}, { 0,  0},
        { 1,  0}, { 0,  1}, { 1,  1}, { 1, -1}
    };
    vector<int>    opp_vect = { 5, 6, 7, 8, 4, 0, 1, 2, 3 };
    vector<double> w_vect   = {
        1./9., 1./9., 1./36., 1./36., 4./9.,
        1./9., 1./9., 1./36., 1./36.
    };
    return make_tuple(c_vect, opp_vect, w_vect);
}

// ───────────────────────── lattice-unit parameters ─────────────────────
// Compute Reynolds, Atwood, capillary and Peclet numbers along with dx and dt.
inline auto lbParameters_Porous2D_MRT(double ulb, int lref, double mu_l, double Sigma,
                                      double rho_l, double rho_g, double M)
{
    double Re   = rho_l * ulb * static_cast<double>(lref) / mu_l;
    double At   = (rho_l - rho_g ) / (rho_l + rho_g);
    double Ca   = mu_l * ulb / Sigma;
    double Pe   = ulb * static_cast<double>(lref) / M;
    double dx   = 1.0 / static_cast<double>(lref);
    double dt   = dx * ulb;
    return make_tuple(Re, At, Ca, Pe, dx, dt);
}

// Print simulation parameters.
inline void printParameters_Porous2D_MRT(const Dim_Porous2D_MRT &dim,
                                         double Re, double At, double Ca, double Pe, double ulb,
                                         int N, double max_t)
{
    cout << "Porous 2-D (Conservative PF — MRT)" << '\n'
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

// ───────────────────────── timing helpers ──────────────────────────────
inline auto restartClock_Porous2D_MRT()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_Porous2D_MRT(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    int dure = static_cast<int>(us.count());
    cout << "Runtime: " << dure/1000000. << " seconds " << endl;
    cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

// ───────────────────────── LBM functor (MRT) ──────────────────────────
// The class implements the conservative phase‑field LBM with MRT for porous media.
struct LBM_Porous2D_MRT
{
    using  CellData = double;
    static constexpr int Q = 9;
    static constexpr double cs2 = 1.0/3.0;
    static constexpr double PI  = 3.1415926535897932384626433832795;

    // raw buffers (allocated externally)
    CellData*          lattice;
    CellType_Porous2D_MRT* flag;
    int*               parity;
    std::array<int,2>* c;
    int*               opp;
    double*            w;

    // neighbor tables (precomputed)
    int* nb    = nullptr; // size = nelem*Q
    int* nbb   = nullptr; // mirror index (used for bounce‑back)

    // precomputed wall normals (geometry-based)
    double* wall_nx = nullptr;  // size = nelem
    double* wall_ny = nullptr;  // size = nelem
    // cached φ (scalar) per cell per step
    double* phi_cache = nullptr; // size = nelem

    // force tracking (for debugging/analysis)
    double* Fs_x = nullptr;  // viscous force x-component
    double* Fs_y = nullptr;  // viscous force y-component
    double* Fb_x = nullptr;  // body force x-component
    double* Fb_y = nullptr;  // body force y-component
    double* Fp_x = nullptr;  // pressure gradient force x-component
    double* Fp_y = nullptr;  // pressure gradient force y-component

    // model params
    double phi_l;
    double phi_g;
    double rho_l;
    double rho_g;
    double mu_l;
    double mu_g;
    double Sigma;
    double Fbx;
    double Fby;
    double xi;
    double M;
    double contact_angle_deg;
    bool   natural_wetting;
    Dim_Porous2D_MRT dim;

    // injection parameters
    double u_inj;
    double inj_radius_px;

    bool has_walls = true;

    // hydrodynamic MRT data (Fakhari et al. Eq. (27–31))
    bool   use_MRT          = true;      // switch MRT on/off
    bool   use_tau_linear   = false;     // false → Eq. (19), true → linear/harmonic interpolation
    double s_nu             = 1.0;       // shear relaxation (0 < s_nu < 2)
    // diagonal entries of S in moment space (will be updated)
    double S_diag[Q];
    // moment transform matrices
    double Mmat[Q][Q];
    double Minv[Q][Q];
    // mapping between custom and standard moment ordering
    static constexpr int custom_to_std[Q] = {3, 4, 7, 6, 0, 1, 2, 5, 8};
    static constexpr int std_to_custom[Q]  = {4, 5, 6, 0, 1, 7, 3, 2, 8};

    // ───────── index helpers ─────────
    inline size_t xyz_to_i (int x, int y) const {
        return static_cast<size_t>(y) + static_cast<size_t>(dim.ny) * static_cast<size_t>(x);
    }
    inline auto i_to_xyz (int i) const {
        int iX = i / dim.ny;
        int iY = i % dim.ny;
        return std::make_tuple(iX, iY);
    }

    // ───────── array accessors (ping–pong) ───────
    inline double& f   (int i,int k)        { return lattice[              k*dim.nelem + i]; }
    inline double& fin (int i,int k)        { return lattice[*parity  *dim.npop + k*dim.nelem + i]; }
    inline double& fout(int i,int k)        { return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }
    inline double& g   (int i,int k)        { return lattice[2*dim.npop                  + k*dim.nelem + i]; }
    inline double& gin (int i,int k)        { return lattice[*parity  *dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    inline double& gout(int i,int k)        { return lattice[(1-*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    inline double  fin (int i,int k)  const { return lattice[*parity  *dim.npop + k*dim.nelem + i]; }
    inline double  gin (int i,int k)  const { return lattice[*parity  *dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    // ───────── neighbor table builder ───────
    inline void build_neighbors()
    {
        for (int x = 0; x < dim.nx; ++x) {
            for (int y = 0; y < dim.ny; ++y) {
                int i = static_cast<int>(xyz_to_i(x,y));
                for (int k=0;k<Q;++k) {
                    int XX  = (x + c[k][0] + dim.nx) % dim.nx;
                    int YY  = (y + c[k][1] + dim.ny) % dim.ny;
                    int XXb = (x - c[k][0] + dim.nx) % dim.nx;
                    int YYb = (y - c[k][1] + dim.ny) % dim.ny;
                    nb [i*Q + k] = static_cast<int>(xyz_to_i(XX ,YY ));
                    nbb[i*Q + k] = static_cast<int>(xyz_to_i(XXb,YYb));
                }
            }
        }
    }

    // ───────── MRT initialization (standard D2Q9 moments) ───────
    inline void init_MRT()
    {
        // Lallemand–Luo M matrix for standard D2Q9 lattice.
        const double Mloc[Q][Q] = {
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
        const double Minv_loc[Q][Q] = {
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
        // copy to member arrays
        for (int i=0;i<Q;++i){
            for (int j=0;j<Q;++j){
                Mmat[i][j] = Mloc[i][j];
                Minv[i][j] = Minv_loc[i][j];
            }
        }
        // diagonal S: all modes relax at unity except shear (indices 7,8)
        for (int a=0;a<Q;++a) S_diag[a] = 1.0;
        // shear relaxation parameter s_nu will be set at run time based on viscosity.
    }

    // ───────── compute phase‑field cache ───────
    // Build phi_cache by summing f-distributions or setting solid/injection values.
    // The tile sizes are chosen automatically based on domain dimensions to avoid
    // manual compile‑time toggles.  We pick tile dimensions as 1/8 of the domain size,
    // but clamp them to be at least 1.
    inline void build_phi_cache()
    {
        int TX = std::max(1, dim.nx / 8);
        int TY = std::max(1, dim.ny / 8);
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range2d<int>(0, dim.nx, TX,
                                              0, dim.ny, TY),
            [&](const oneapi::tbb::blocked_range2d<int>& r){
                for (int x = r.rows().begin(); x < r.rows().end(); ++x) {
                    size_t base = static_cast<size_t>(x) * static_cast<size_t>(dim.ny);
                    for (int y = r.cols().begin(); y < r.cols().end(); ++y) {
                        int i = static_cast<int>(base + static_cast<size_t>(y));
                        if (flag[i] == CellType_Porous2D_MRT::bounce_back ||
                            flag[i] == CellType_Porous2D_MRT::injection   ||
                            flag[i] == CellType_Porous2D_MRT::outlet) {
                            phi_cache[i] = (flag[i] == CellType_Porous2D_MRT::injection) ? phi_l : phi_g;
                            continue;
                        }
                        // sum the nine f populations to get φ (phase field)
                        phi_cache[i] = (fin(i,0)+fin(i,2)+fin(i,3)) +
                                       (fin(i,5)+fin(i,7)+fin(i,8)) +
                                       (fin(i,6)+fin(i,1)+fin(i,4));
                    }
                }
            }
        );
    }

    // ───────── gradient of phase field φ using central differences ───────
    // Implementation uses second‑order isotropic differences【676137786399381†L626-L633】 and
    // bounce‑back when a neighbor is solid.
    inline std::array<double,2> gradient_phi(int i) const {
        double gx=0.0, gy=0.0;
        const int base = i * Q;
        for (int k=0; k<Q; ++k) {
            int j = nb[base + k];
            // if neighbor is solid, use mirror (bounce‑back) cell to preserve symmetry
            if (flag[j] == CellType_Porous2D_MRT::bounce_back)
                j = nbb[base + k];
            double phi_nb = phi_cache[j];
            gx += w[k] * c[k][0] * phi_nb;
            gy += w[k] * c[k][1] * phi_nb;
        }
        return { gx / cs2, gy / cs2 };
    }

    // Laplacian of φ for chemical potential.
    inline double laplacian_phi(int i) const {
        const double phi_c = phi_cache[i];
        double sum = 0.0;
        const int base = i * Q;
        for (int k=0; k<Q; ++k) {
            int j = nb[base + k];
            sum += w[k] * (phi_cache[j] - phi_c);
        }
        return 2.0 * sum / cs2;
    }

    // Macroscopic pressure P* = sum of g populations.
    inline double macro_P_only(int i) const {
        return (gin(i,0)+gin(i,2)+gin(i,3)) +
               (gin(i,5)+gin(i,7)+gin(i,8)) +
               (gin(i,6)+gin(i,1)+gin(i,4));
    }

    // Gradient of density from gradient of φ using constant scaling.
    inline std::array<double,2> gradient_rho_from_gradphi(const std::array<double,2>& grad_phi) const {
        double dr = rho_l - rho_g;
        return { dr*grad_phi[0], dr*grad_phi[1] };
    }

    // Local density and viscosity from φ (linear interpolation).
    inline double local_rho(double phi) const {
        return rho_g + (phi - phi_g) * (rho_l - rho_g);
    }
    inline double local_mu(double phi) const {
        return mu_g + (phi - phi_g) * (mu_l - mu_g);
    }

    // Relaxation time and kinematic viscosity.
    inline double local_tau(double rho,double mu) const { return mu / (rho*cs2); }
    inline double local_nu (double tau) const { return tau * cs2; }

    // Chemical potential μ_φ = δF/δφ (phase‑field term).
    inline double mu_phi(double phi, double lap_phi) const {
        double beta  = 12.0 * Sigma / xi;
        double kappa = 3.0 * Sigma * xi / 2.0;
        double phi0  = 0.5 * (phi_l + phi_g);
        return 4.0 * beta * (phi - phi_g) * (phi - phi_l) * (phi - phi0) - kappa * lap_phi;
    }

    // Viscous force F_s = μ_φ ∇φ.
    inline std::array<double,2> Fs_from(double mu_phi_val,const std::array<double,2>& grad_phi) const {
        return { mu_phi_val * grad_phi[0], mu_phi_val * grad_phi[1] };
    }
    // Body force F_b = ρ F_ext.
    inline std::array<double,2> Fb_from(double rho) const {
        return { rho*Fbx, rho*Fby };
    }
    // Pressure gradient force F_p = -P* c_s^2 ∇ρ.
    inline std::array<double,2> Fp_from(double Pstar,const std::array<double,2>& grad_rho) const {
        return { -Pstar * cs2 * grad_rho[0], -Pstar * cs2 * grad_rho[1] };
    }

    // Macroscopic velocity from momentum and forces.
    inline auto macro_u(int i) const {
        std::array<double,2> u{};
        double Xg_M1 = gin(i, 0) + gin(i, 2) + gin(i, 3);
        double Xg_P1 = gin(i, 5) + gin(i, 7) + gin(i, 8);

        double Yg_P1 = gin(i, 3) + gin(i, 7) + gin(i, 6);
        double Yg_M1 = gin(i, 2) + gin(i, 1) + gin(i, 8);

        u[0] = Xg_P1 - Xg_M1;
        u[1] = Yg_P1 - Yg_M1;
        return u;
    }

    // Hydrodynamic forcing term F_μ derived from MRT stress.
    inline std::array<double,2> Fmu_from(int i,
                                         const std::array<double,2>& u,
                                         double Pstar,double tau,double nu,
                                         const std::array<double,2>& grad_rho) const
    {
        // compute u²/(2c_s²)
        double u_sq = (u[0]*u[0] + u[1]*u[1]) / (2.0*cs2);
        // local pointers for weight vector
        const double* wt = w;
        // gather current g distributions and equilibrium in standard ordering
        double gvec_std[Q];
        double geq_std[Q];
        for (int k=0; k<Q; ++k) {
            int s = custom_to_std[k];
            // current distribution
            double gk = gin(i,k);

            double ex = static_cast<double>(c[k][0]);
            double ey = static_cast<double>(c[k][1]);
            double ck_u = ex*u[0] + ey*u[1];
            double gamma_k = wt[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - u_sq);
            double geq_k   = Pstar * wt[k] + (gamma_k - wt[k]);
            gvec_std[s] = gk;
            geq_std[s]  = geq_k;
        }
        // form moment vector m = M*(g - g_eq)
        double m_vec[Q];
        for (int r=0; r<Q; ++r) {
            double sum = 0.0;
            for (int a=0; a<Q; ++a) {
                sum += Mmat[r][a] * (gvec_std[a] - geq_std[a]);
            }
            m_vec[r] = sum;
        }
        // relaxation diagonal: shear modes (7,8) depend on tau
        double s_nu_val = 1.0 / (tau + 0.5);
        double Sdiag[Q];
        for (int r=0; r<Q; ++r) Sdiag[r] = 1.0;
        Sdiag[7] = s_nu_val;
        Sdiag[8] = s_nu_val;
        // post‑collision moments s_m = Sdiag * m
        double s_m[Q];
        for (int r=0; r<Q; ++r) {
            s_m[r] = Sdiag[r] * m_vec[r];
        }
        // back to velocity space: A_std = M_inv * s_m
        double A_std[Q];
        for (int a=0; a<Q; ++a) {
            double sum = 0.0;
            for (int r=0; r<Q; ++r) {
                sum += Minv[a][r] * s_m[r];
            }
            A_std[a] = sum;
        }
        // reorder to custom ordering
        double A_custom[Q];
        for (int k=0; k<Q; ++k) {
            int s = custom_to_std[k];
            A_custom[k] = A_std[s];
        }
        // contract stress tensor with gradient of density using custom c and A_custom
        double Txx = 0.0, Txy = 0.0, Tyy = 0.0;
        for (int beta=0; beta<Q; ++beta) {
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

    // Macroscopic velocity with forcing (trapezoidal rule).
    inline std::array<double,2> velocity_from(const std::array<double,2>& u_macro,
                                              const std::array<double,2>& F_total,
                                              double rho) const
    {
        return { u_macro[0] + F_total[0]/(2.0*rho),
                 u_macro[1] + F_total[1]/(2.0*rho) };
    }

    // Phase‑field mobility forcing.
    inline std::array<double,2> Fphi_vec(double phi,const std::array<double,2>& grad_phi) const {
        double phi0 = 0.5*(phi_g + phi_l);
        double fac  = M / cs2 * (4.0*phi*(1.0 - phi)) / xi;
        double s   = std::sqrt(grad_phi[0]*grad_phi[0] + grad_phi[1]*grad_phi[1]) + 1e-32;
        return { fac*grad_phi[0]/s, fac*grad_phi[1]/s };
    }

    // Stream to neighbor or bounce‑back.
    inline void stream_tab(int i,int k,double pf,double pg)
    {
        if (!has_walls) {
            int j = nb[i*Q + k];
            fout(j, k) = pf;
            gout(j, k) = pg;
            return;
        }
        int j = nb[i*Q + k];
        if (flag[j] == CellType_Porous2D_MRT::bounce_back) {
            // bounce back: reflect into opposite direction
            int kb = opp[k];
            fout(i, kb) = pf;
            gout(i, kb) = pg;
        } else {
            fout(j, k) = pf;
            gout(j, k) = pg;
        }
    }

    // ───────── BGK collision for f (phase field) and g (pressure) ───────
    inline std::tuple<double,double,double,double>
    collideBgk_f_and_g(int i,int k,
                       const std::array<double,2>& u,double rho,
                       double phi, double P, double usqr, double tau,
                       const std::array<double,2>& F,
                       const std::array<double,2>& F_phi)
    {
        double ck_u = c[k][0]*u[0] + c[k][1]*u[1];
        double tau_phi = M / cs2;

        double gamma      = w[k] * (1.0 + ck_u / cs2 + ck_u*ck_u / (2.0 * cs2 * cs2) - usqr);
        double gamma_opp  = w[opp[k]] * (1.0 - ck_u / cs2 + ck_u*ck_u / (2.0 * cs2 * cs2) - usqr);
        double geq        = P * w[k] + (gamma - w[k]);
        double geq_opp    = P * w[opp[k]] + (gamma_opp - w[opp[k]]);

        double F_f        = w[k] * (F_phi[0] * c[k][0] + F_phi[1] * c[k][1]);

        double feq_bar      = phi * gamma      - F_f / 2.0;
        double feq_bar_opp  = phi * gamma_opp  + F_f / 2.0;

        double pf     = fin(i,k)      - (fin(i,k)      - feq_bar     ) / (0.5 + tau_phi) + F_f;
        double pf_opp = fin(i,opp[k]) - (fin(i,opp[k]) - feq_bar_opp ) / (0.5 + tau_phi) - F_f;

        double pg    = gin(i,k);
        double pg_opp= gin(i,opp[k]);

        return {pf, pg, pf_opp, pg_opp};
    }

    // ───────────────────────── per-cell update ─────────────────────────
    // This functor performs the collision and streaming at a given lattice cell.
    inline void operator()(double& f0)
    {
        int i = static_cast<int>(&f0 - lattice);
        if (flag[i] == CellType_Porous2D_MRT::bounce_back ||
            flag[i] == CellType_Porous2D_MRT::injection   ||
            flag[i] == CellType_Porous2D_MRT::outlet) {
            return;
        }
        // phase field, pressure and densities
        double phi   = phi_cache[i];
        double Pstar = macro_P_only(i);
        double rho   = local_rho(phi);
        double mu    = local_mu(phi);
        double tau   = local_tau(rho, mu);
        double nu    = local_nu(tau);
        // compute macroscopic velocity and gradients
        auto  u_macro   = macro_u(i);
        auto  grad_phi  = gradient_phi(i);
        double lap_phi  = laplacian_phi(i);
        auto  grad_rho  = gradient_rho_from_gradphi(grad_phi);

        double mu_phi_val = mu_phi(phi, lap_phi);
        auto   F_phi      = Fphi_vec(phi, grad_phi);
        // compute Fs, Fb, Fp without Fμ
        auto Fs = Fs_from(mu_phi_val, grad_phi);
        auto Fb = Fb_from(rho);
        auto Fp = Fp_from(Pstar, grad_rho);
        std::array<double,2> F_no_mu { Fs[0] + Fb[0] + Fp[0], Fs[1] + Fb[1] + Fp[1] };

        // first estimate u using Fs+Fb+Fp
        auto u_first = velocity_from(u_macro, F_no_mu, rho);
        // midpoint velocity for Fμ calculation
        std::array<double,2> u_mid { 0.5*(u_macro[0] + u_first[0]), 0.5*(u_macro[1] + u_first[1]) };
        auto Fmu_vec = Fmu_from(i, u_mid, Pstar, tau, nu, grad_rho);

        // total forcing
        std::array<double,2> F_total{
            Fs[0] + Fb[0] + Fp[0] + Fmu_vec[0],
            Fs[1] + Fb[1] + Fp[1] + Fmu_vec[1]
        };
        // final velocity with all forces
        auto   u    = velocity_from(u_macro, F_total, rho);
        double usqr = (u[0]*u[0] + u[1]*u[1]) / ( 2.0 * cs2 );

        // (1) hydrodynamic MRT collision for g populations
        double gvec_std[Q];
        double geq_std[Q];
        double Fg_std[Q];
        for (int k=0; k<Q; ++k) {
            int s = custom_to_std[k];
            double ex = static_cast<double>(c[k][0]);
            double ey = static_cast<double>(c[k][1]);
            double ck_u = ex*u[0] + ey*u[1];
            double gamma_k = w[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - usqr);
            double g_k  = gin(i,k);
            double geq_k= Pstar * w[k] + (gamma_k - w[k]);
            double F_k  = w[k] * (F_total[0] * ex + F_total[1] * ey) / (rho * cs2);
            gvec_std[s] = g_k;
            geq_std[s]  = geq_k;
            Fg_std[s]   = F_k;
        }
        // moment space arrays
        double m_vec[Q] = {0.0}, m_eq[Q] = {0.0}, Fm[Q] = {0.0}, m_post[Q] = {0.0};
        for (int r=0; r<Q; ++r) {
            double sum = 0.0, sum_eq = 0.0, sum_f = 0.0;
            for (int a=0; a<Q; ++a) {
                sum    += Mmat[r][a] * gvec_std[a];
                sum_eq += Mmat[r][a] * geq_std[a];
                sum_f  += Mmat[r][a] * Fg_std[a];
            }
            m_vec[r] = sum;
            m_eq[r]  = sum_eq;
            Fm[r]    = sum_f;
        }
        // relaxation diagonal: shear modes use s_nu
        double s_nu_val = 1.0 / (tau + 0.5);
        double Sdiag[Q];
        for (int r=0; r<Q; ++r) {
            Sdiag[r] = 1.0;
        }
        Sdiag[7] = s_nu_val;
        Sdiag[8] = s_nu_val;
        for (int r=0; r<Q; ++r) {
            double Sr = Sdiag[r];
            m_post[r] = m_vec[r] - Sr * (m_vec[r] - m_eq[r]) + (1.0 - 0.5 * Sr) * Fm[r];
        }
        // back to velocity space
        double g_post_std[Q];
        for (int a=0; a<Q; ++a) {
            double sum = 0.0;
            for (int r=0; r<Q; ++r) {
                sum += Minv[a][r] * m_post[r];
            }
            g_post_std[a] = sum;
        }
        // reorder to custom ordering
        double g_post[Q];
        for (int k=0; k<Q; ++k) {
            int s = custom_to_std[k];
            g_post[k] = g_post_std[s];
        }
        // perform streaming for g
        for (int k=0; k<Q; ++k) {
            int j = nb[i*Q + k];
            if (flag[j] == CellType_Porous2D_MRT::bounce_back) {
                // bounce back: reflect into opposite direction at current node
                int kb = opp[k];
                gout(i,kb) = g_post[k];
            } else {
                gout(j,k) = g_post[k];
            }
        }

        // (2) BGK for φ (f-lattice) + streaming
        for (int k = 0; k < 4; ++k) {
            auto [pf, pg, pf_op, pg_op] = collideBgk_f_and_g(
                i, k, u, rho, phi, Pstar, usqr, tau, F_total, F_phi
            );
            stream_tab(i, k     , pf   , pg   );
            stream_tab(i, opp[k], pf_op, pg_op);
        }
        // Rest population for f, simple BGK
        {
            int k = 4;
            double tau_phi = M / cs2;
            double feq0 = phi * w[k] * (1.0 - usqr);
            fout(i,k) = fin(i,k) - (fin(i,k) - feq0)  / (tau_phi + 0.5);
        }
    }

    // ───────── initialization ─────────
    inline void iniLattice(double& f0)
    {
        int i = static_cast<int>(&f0 - lattice);
        if (flag[i] == CellType_Porous2D_MRT::bounce_back) {
            for (int k = 0; k < Q; ++k) { fin(i,k)=0.0; gin(i,k)=0.0; }
            return;
        }
        double phi = (flag[i] == CellType_Porous2D_MRT::injection) ? phi_l : phi_g;
        for (int k = 0; k < Q; ++k) {
            fin(i,k) = phi * w[k];
            gin(i,k) = 0.0;
        }
    }

    // ───────── compute wall normals (geometry-based) ───────
    inline void compute_wall_normals()
    {
        const int nx = dim.nx;
        const int ny = dim.ny;

        for (size_t i = 0; i < dim.nelem; ++i) {
            wall_nx[i] = 0.0;
            wall_ny[i] = 0.0;
        }
        auto idx = [this](int x, int y) -> int {
            return static_cast<int>(xyz_to_i(x, y));
        };
        auto is_fluid = [this, nx, ny, &idx](int x, int y) -> bool {
            if (x < 0 || x >= nx || y < 0 || y >= ny) return false;
            int j = idx(x, y);
            return (flag[j] != CellType_Porous2D_MRT::bounce_back);
        };
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                int i = idx(x, y);
                if (flag[i] != CellType_Porous2D_MRT::bounce_back) continue;

                bool   has_fluid = false;
                double nx_acc = 0.0;
                double ny_acc = 0.0;

                // Cardinals
                if (is_fluid(x+1, y)) { nx_acc += 1.0; has_fluid = true; }
                if (is_fluid(x-1, y)) { nx_acc -= 1.0; has_fluid = true; }
                if (is_fluid(x, y+1)) { ny_acc += 1.0; has_fluid = true; }
                if (is_fluid(x, y-1)) { ny_acc -= 1.0; has_fluid = true; }

                // Diagonals if no cardinal neighbor
                if (!has_fluid) {
                    if (is_fluid(x+1, y+1)) { nx_acc += 1.0; ny_acc += 1.0; has_fluid = true; }
                    if (is_fluid(x-1, y+1)) { nx_acc -= 1.0; ny_acc += 1.0; has_fluid = true; }
                    if (is_fluid(x+1, y-1)) { nx_acc += 1.0; ny_acc -= 1.0; has_fluid = true; }
                    if (is_fluid(x-1, y-1)) { nx_acc -= 1.0; ny_acc -= 1.0; has_fluid = true; }
                }
                if (!has_fluid) continue;
                double len = std::sqrt(nx_acc*nx_acc + ny_acc*ny_acc);
                if (len > 0.0) {
                    wall_nx[i] = nx_acc / len;
                    wall_ny[i] = ny_acc / len;
                }
            }
        }
    }

    // ───────── helper: close diagonal corners ───────────────────────
    inline void close_solid_corners_Porous2D_MRT(LBM_Porous2D_MRT& lbm)
    {
        const auto& diml = lbm.dim;
        auto idx = [&](int X,int Y){
            int xw = (X % diml.nx + diml.nx) % diml.nx;      // wrap X
            return static_cast<size_t>(lbm.xyz_to_i(xw, Y));
        };
        std::vector<uint8_t> add_solid(diml.nelem, 0);
        for (int y = 1; y < diml.ny-1; ++y) {
            for (int x = 0; x < diml.nx; ++x) {
                size_t i = lbm.xyz_to_i(x,y);
                if (lbm.flag[i] == CellType_Porous2D_MRT::bounce_back) continue;
                bool sL = lbm.flag[idx(x-1,y  )] == CellType_Porous2D_MRT::bounce_back;
                bool sR = lbm.flag[idx(x+1,y  )] == CellType_Porous2D_MRT::bounce_back;
                bool sD = lbm.flag[idx(x  ,y-1)] == CellType_Porous2D_MRT::bounce_back;
                bool sU = lbm.flag[idx(x  ,y+1)] == CellType_Porous2D_MRT::bounce_back;
                bool sUL= lbm.flag[idx(x-1,y+1)] == CellType_Porous2D_MRT::bounce_back;
                bool sUR= lbm.flag[idx(x+1,y+1)] == CellType_Porous2D_MRT::bounce_back;
                bool sDL= lbm.flag[idx(x-1,y-1)] == CellType_Porous2D_MRT::bounce_back;
                bool sDR= lbm.flag[idx(x+1,y-1)] == CellType_Porous2D_MRT::bounce_back;
                if ((sU && sR && sUR) || (sU && sL && sUL) ||
                    (sD && sR && sDR) || (sD && sL && sDL)) {
                    add_solid[i] = 1;
                }
            }
        }
        for (size_t i = 0; i < add_solid.size(); ++i) {
            if (!add_solid[i]) continue;
            lbm.flag[i] = CellType_Porous2D_MRT::bounce_back;
            for (int k = 0; k < Q; ++k) {
                lbm.fin(static_cast<int>(i),k) = 0.0;
                lbm.gin(static_cast<int>(i),k) = 0.0;
            }
        }
    }

    // ───────── Injection + outlet BCs (Zou-He for outlets) ────────────────────
    inline void apply_boundary_conditions_Porous2D_MRT(LBM_Porous2D_MRT& lbm)
    {
        const int nx = lbm.dim.nx;
        const int ny = lbm.dim.ny;
        // 1) Injection: homogeneous injection with uniform velocity (upward direction)
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int i = static_cast<int>(lbm.xyz_to_i(x,y));
                if (lbm.flag[i] == CellType_Porous2D_MRT::injection) {
                    // Uniform velocity: upward (positive y direction)
                    double ux = 0.0;
                    double uy = lbm.u_inj;
                    const double u2  = (ux*ux + uy*uy) / (2.0*cs2);
                    for (int k = 0; k < Q; ++k) {
                        const double cu    = lbm.c[k][0]*ux + lbm.c[k][1]*uy;
                        const double gamma = lbm.w[k]*(1.0 + cu/cs2 + (cu*cu)/(2.0*cs2*cs2) - u2);
                        lbm.gout(i,k) = gamma;
                        lbm.fout(i,k) = lbm.phi_l * gamma;
                    }
                }
            }
        }
        // 2) Zou-He constant pressure outlets: zero pressure
        double P_outlet = 0.0;
        // top edge
        for (int x = 0; x < nx; ++x) {
            int iT = static_cast<int>(lbm.xyz_to_i(x, ny-1));
            int jT = static_cast<int>(lbm.xyz_to_i(x, ny-2));
            if (lbm.flag[iT] == CellType_Porous2D_MRT::outlet) {
                for (int k = 0; k < Q; ++k) {
                    double f_known = lbm.gout(jT, k);
                    lbm.gout(iT, k) = f_known - 2.0 * lbm.w[k] * P_outlet * lbm.c[k][1];
                }
                for (int k = 0; k < Q; ++k) {
                    lbm.fout(iT, k) = lbm.fout(jT, k);
                }
            }
        }
        // bottom edge
        for (int x = 0; x < nx; ++x) {
            int iB = static_cast<int>(lbm.xyz_to_i(x, 0));
            int jB = static_cast<int>(lbm.xyz_to_i(x, 1));
            if (lbm.flag[iB] == CellType_Porous2D_MRT::outlet) {
                for (int k = 0; k < Q; ++k) {
                    double f_known = lbm.gout(jB, k);
                    lbm.gout(iB, k) = f_known - 2.0 * lbm.w[k] * P_outlet * lbm.c[k][1];
                }
                for (int k = 0; k < Q; ++k) {
                    lbm.fout(iB, k) = lbm.fout(jB, k);
                }
            }
        }
        // left edge
        for (int y = 0; y < ny; ++y) {
            int iL = static_cast<int>(lbm.xyz_to_i(0, y));
            int jL = static_cast<int>(lbm.xyz_to_i(1, y));
            if (lbm.flag[iL] == CellType_Porous2D_MRT::outlet) {
                for (int k = 0; k < Q; ++k) {
                    double f_known = lbm.gout(jL, k);
                    lbm.gout(iL, k) = f_known - 2.0 * lbm.w[k] * P_outlet * lbm.c[k][0];
                }
                for (int k = 0; k < Q; ++k) {
                    lbm.fout(iL, k) = lbm.fout(jL, k);
                }
            }
        }
        // right edge
        for (int y = 0; y < ny; ++y) {
            int iR = static_cast<int>(lbm.xyz_to_i(nx-1, y));
            int jR = static_cast<int>(lbm.xyz_to_i(nx-2, y));
            if (lbm.flag[iR] == CellType_Porous2D_MRT::outlet) {
                for (int k = 0; k < Q; ++k) {
                    double f_known = lbm.gout(jR, k);
                    lbm.gout(iR, k) = f_known - 2.0 * lbm.w[k] * P_outlet * lbm.c[k][0];
                }
                for (int k = 0; k < Q; ++k) {
                    lbm.fout(iR, k) = lbm.fout(jR, k);
                }
            }
        }
    }

    // ───────── VTK output ────────────────────────────────
    inline void saveVtkFields_Porous2D_MRT(LBM_Porous2D_MRT& lbm, int time_iter, double dx = 0.) {
        Dim_Porous2D_MRT const& diml = lbm.dim;
        if (dx == 0.0) dx = 1.0 / diml.nx;
        int dimZ = 1;
        stringstream ss;
        ss << "sol_" << setw(7) << setfill('0') << time_iter << ".vtk";
        ofstream os(ss.str());
        os << "# vtk DataFile Version 2.0\n";
        os << "iteration " << time_iter << "\n";
        os << "ASCII\n\n";
        os << "DATASET STRUCTURED_POINTS\n";
        os << "DIMENSIONS " << diml.nx << " " << diml.ny << " " << dimZ << "\n";
        os << "ORIGIN 0 0 0\n";
        os << "SPACING " << dx << " " << dx << " " << dx << "\n\n";
        os << "POINT_DATA " << diml.nx * diml.ny * dimZ << "\n";

        // φ
        os << "SCALARS phi float 1\n";
        os << "LOOKUP_TABLE default\n";
        for (int y = 0; y < diml.ny; ++y) {
            for (int x = 0; x < diml.nx; ++x) {
                int idxc = static_cast<int>(lbm.xyz_to_i(x, y));
                float val = 0.f;
                if (lbm.flag[idxc] == CellType_Porous2D_MRT::bulk ||
                    lbm.flag[idxc] == CellType_Porous2D_MRT::outlet ||
                    lbm.flag[idxc] == CellType_Porous2D_MRT::injection) {
                    val = static_cast<float>(lbm.phi_cache[idxc]);
                }
                os << val << " ";
            }
            os << "\n";
        }
        os << "\n";

        // ρ
        os << "SCALARS density float 1\n";
        os << "LOOKUP_TABLE default\n";
        for (int y = 0; y < diml.ny; ++y) {
            for (int x = 0; x < diml.nx; ++x) {
                int idxc = static_cast<int>(lbm.xyz_to_i(x, y));
                float val = 0.f;
                if (lbm.flag[idxc] == CellType_Porous2D_MRT::bulk ||
                    lbm.flag[idxc] == CellType_Porous2D_MRT::outlet ||
                    lbm.flag[idxc] == CellType_Porous2D_MRT::injection) {
                    val = static_cast<float>(lbm.local_rho(lbm.phi_cache[idxc]));
                }
                os << val << " ";
            }
            os << "\n";
        }
        os << "\n";

        // p* * rho / 3
        os << "SCALARS pressure float 1\n";
        os << "LOOKUP_TABLE default\n";
        for (int y = 0; y < diml.ny; ++y) {
            for (int x = 0; x < diml.nx; ++x) {
                int idxc = static_cast<int>(lbm.xyz_to_i(x, y));
                double val = 0.0;
                if (lbm.flag[idxc] == CellType_Porous2D_MRT::bulk ||
                    lbm.flag[idxc] == CellType_Porous2D_MRT::outlet ||
                    lbm.flag[idxc] == CellType_Porous2D_MRT::injection) {
                    double P   = lbm.macro_P_only(idxc);
                    double rho = lbm.local_rho(lbm.phi_cache[idxc]);
                    val = P * rho / 3.0;
                }
                os << val << " ";
            }
            os << "\n";
        }
        os << "\n";

        // Velocity
        os << "VECTORS velocity float\n";
        for (int y = 0; y < diml.ny; ++y) {
            for (int x = 0; x < diml.nx; ++x) {
                int idxc = static_cast<int>(lbm.xyz_to_i(x, y));
                double ux = 0.0, uy = 0.0;
                if (lbm.flag[idxc] == CellType_Porous2D_MRT::bulk ||
                    lbm.flag[idxc] == CellType_Porous2D_MRT::outlet ||
                    lbm.flag[idxc] == CellType_Porous2D_MRT::injection) {
                    double rho = lbm.local_rho(lbm.phi_cache[idxc]);
                    auto   u_macro  = lbm.macro_u(idxc);
                    auto   grad_phi = lbm.gradient_phi(idxc);
                    double lap_phi  = lbm.laplacian_phi(idxc);
                    auto   grad_rho = lbm.gradient_rho_from_gradphi(grad_phi);
                    double mu_phi_val = lbm.mu_phi(lbm.phi_cache[idxc], lap_phi);
                    auto   Fs     = lbm.Fs_from(mu_phi_val, grad_phi);
                    auto   Fb     = lbm.Fb_from(rho);
                    double P      = lbm.macro_P_only(idxc);
                    auto   Fp     = lbm.Fp_from(P, grad_rho);
                    std::array<double,2> Ftot { Fs[0] + Fb[0] + Fp[0], Fs[1] + Fb[1] + Fp[1] };
                    auto u_est = lbm.velocity_from(u_macro, Ftot, rho);
                    ux = u_est[0]; uy = u_est[1];
                }
                os << static_cast<float>(ux) << " " << static_cast<float>(uy) << " 0\n";
            }
        }
        os << "\n";

        // Flag
        os << "SCALARS Flag int 1\n";
        os << "LOOKUP_TABLE default\n";
        for (int y = 0; y < diml.ny; ++y) {
            for (int x = 0; x < diml.nx; ++x) {
                size_t idxc = lbm.xyz_to_i(x, y);
                int flag_int = 0;
                if      (lbm.flag[idxc] == CellType_Porous2D_MRT::bounce_back) flag_int = 1;
                else if (lbm.flag[idxc] == CellType_Porous2D_MRT::injection)   flag_int = 2;
                else if (lbm.flag[idxc] == CellType_Porous2D_MRT::outlet)      flag_int = 3;
                os << flag_int << " ";
            }
            os << "\n";
        }
        os << "\n";

        // Wall normals (optional)
        os << "VECTORS wall_normal float\n";
        for (int y = 0; y < diml.ny; ++y) {
            for (int x = 0; x < diml.nx; ++x) {
                int idxc = static_cast<int>(lbm.xyz_to_i(x, y));
                float nxw = 0.f, nyw = 0.f;
                if (lbm.flag[idxc] == CellType_Porous2D_MRT::bounce_back) {
                    nxw = static_cast<float>(lbm.wall_nx[idxc]);
                    nyw = static_cast<float>(lbm.wall_ny[idxc]);
                }
                os << nxw << " " << nyw << " 0\n";
            }
        }
        os << "\n";
        // Force tracking (if allocated)
        if (lbm.Fs_x != nullptr) {
            os << "VECTORS force_viscous float\n";
            for (int y = 0; y < diml.ny; ++y) {
                for (int x = 0; x < diml.nx; ++x) {
                    int idxc = static_cast<int>(lbm.xyz_to_i(x, y));
                    float fsx = 0.f, fsy = 0.f;
                    if (lbm.flag[idxc] != CellType_Porous2D_MRT::bounce_back) {
                        fsx = static_cast<float>(lbm.Fs_x[idxc]);
                        fsy = static_cast<float>(lbm.Fs_y[idxc]);
                    }
                    os << fsx << " " << fsy << " 0\n";
                }
            }
            os << "\n";
            os << "VECTORS force_body float\n";
            for (int y = 0; y < diml.ny; ++y) {
                for (int x = 0; x < diml.nx; ++x) {
                    int idxc = static_cast<int>(lbm.xyz_to_i(x, y));
                    float fbx = 0.f, fby = 0.f;
                    if (lbm.flag[idxc] != CellType_Porous2D_MRT::bounce_back) {
                        fbx = static_cast<float>(lbm.Fb_x[idxc]);
                        fby = static_cast<float>(lbm.Fb_y[idxc]);
                    }
                    os << fbx << " " << fby << " 0\n";
                }
            }
            os << "\n";
            os << "VECTORS force_pressure float\n";
            for (int y = 0; y < diml.ny; ++y) {
                for (int x = 0; x < diml.nx; ++x) {
                    int idxc = static_cast<int>(lbm.xyz_to_i(x, y));
                    float fpx = 0.f, fpy = 0.f;
                    if (lbm.flag[idxc] != CellType_Porous2D_MRT::bounce_back) {
                        fpx = static_cast<float>(lbm.Fp_x[idxc]);
                        fpy = static_cast<float>(lbm.Fp_y[idxc]);
                    }
                    os << fpx << " " << fpy << " 0\n";
                }
            }
            os << "\n";
        }
    }

    // ───────── Energy computation ─────────────────────────
    inline double computeEnergy_Porous2D_MRT(LBM_Porous2D_MRT& lbm) {
        const auto& diml = lbm.dim;
        double energy = 0.0;
        for (int y = 0; y < diml.ny; ++y) {
            for (int x = 0; x < diml.nx; ++x) {
                int idxc = static_cast<int>(lbm.xyz_to_i(x, y));
                if (lbm.flag[idxc] == CellType_Porous2D_MRT::bulk) {
                    auto u_macro  = lbm.macro_u(idxc);
                    double rho = lbm.local_rho(lbm.phi_cache[idxc]);
                    auto u_est = lbm.velocity_from(u_macro, {0.0,0.0}, rho);
                    energy += u_est[0]*u_est[0] + u_est[1]*u_est[1];
                }
            }
        }
        return 0.5 * energy;
    }

    // ───────── Wetting BC on φ_s (Eq. 25) ──────────────────
    // Applies Eq. (25) on all solid wall nodes using wall normals and φ_f.
    inline void apply_wetting_bc_phi_Porous2D_MRT(LBM_Porous2D_MRT& lbm)
    {
        const auto& diml = lbm.dim;
        // Free-energy parameters
        double beta  = 12.0 * lbm.Sigma / lbm.xi;
        double kappa = 3.0 * lbm.Sigma * lbm.xi / 2.0;
        double factor = std::sqrt(2.0 * beta / kappa);
        double theta_rad  = lbm.contact_angle_deg * PI / 180.0;
        double cos_theta  = std::cos(theta_rad);
        for (int x = 0; x < diml.nx; ++x) {
            for (int y = 0; y < diml.ny; ++y) {
                int i = static_cast<int>(lbm.xyz_to_i(x,y));
                if (lbm.flag[i] != CellType_Porous2D_MRT::bounce_back) continue;
                double nxw = lbm.wall_nx[i];
                double nyw = lbm.wall_ny[i];
                if (std::abs(nxw) < 1e-14 && std::abs(nyw) < 1e-14) continue;
                // Find fluid neighbor along normal direction and compute φ_s using Eq. (25).
                int base = i * Q;
                int best_k = -1;
                double best_dp = -1e20;
                for (int k = 0; k < Q; ++k) {
                    int j = lbm.nb[base + k];
                    if (lbm.flag[j] == CellType_Porous2D_MRT::bounce_back) continue;
                    double ex = static_cast<double>(lbm.c[k][0]);
                    double ey = static_cast<double>(lbm.c[k][1]);
                    double dp = ex*nxw + ey*nyw;
                    if (dp > best_dp) {
                        best_dp = dp;
                        best_k = k;
                    }
                }
                if (best_k < 0) continue;
                int j = lbm.nb[base + best_k];
                double phi_f = lbm.phi_cache[j];
                double phi_s;
                if (lbm.natural_wetting) {
                    phi_s = phi_f; // neutral wetting: φ_s = φ_f
                } else {
                    // general contact angle: approximate φ_s from contact angle condition.
                    double dphi_dn = - factor * cos_theta;
                    phi_s = phi_f - dphi_dn;
                }
                lbm.phi_cache[i] = phi_s;
            }
        }
    }
};

// ───────────────────────── helper: read porous mask ─────────────────────
inline void read_Porous2D_MRT_dat(const std::string& fname,
                                  std::vector<CellType_Porous2D_MRT>& flags,
                                  int& nx, int& ny)
{
    std::ifstream f(fname);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open geometry file: " + fname);
    }
    f >> nx >> ny;
    if (!f.good()) {
        throw std::runtime_error("Invalid geometry header in: " + fname);
    }
    flags.assign(static_cast<size_t>(nx) * static_cast<size_t>(ny), CellType_Porous2D_MRT::bulk);
    int val;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (!(f >> val)) {
                throw std::runtime_error("Not enough geometry data in " + fname);
            }
            int yy = ny - 1 - y;  // row 0 = top
            size_t idx = static_cast<size_t>(yy) + static_cast<size_t>(ny) * static_cast<size_t>(x);
            flags[idx] = (val == 1) ? CellType_Porous2D_MRT::bounce_back
                                    : CellType_Porous2D_MRT::bulk;
        }
    }
}

// ───────────────────────── simple geometry: free injection on surface ─────────────────────
inline void init_simple_Porous2D_MRT_geometry(std::vector<CellType_Porous2D_MRT>& flags,
                                              int nx, int ny)
{
    // All cells start as bulk (fluid)
    flags.assign(static_cast<size_t>(nx)*static_cast<size_t>(ny), CellType_Porous2D_MRT::bulk);
}


// ───────────────────────── main driver (Porous 2‑D, MRT) ────────────────
// This function runs a full porous‑media simulation.  It reads
// configuration parameters from ``../apps/Config_Files/config_porous2D.txt``
// (if present), sets up a simple square domain populated with fluid
// everywhere, carves out a circular injection region at the centre,
// applies outlet boundary conditions on the domain edges and then
// advances the solution using a collision–streaming loop.  All
// functions and types use the ``Porous2D_MRT`` naming convention.


inline void Porous2D_MRT()
{
    using namespace std;
    // --------------------------------------------------------------------
    // 1. Open configuration file and set default parameters
    ifstream contfile("../apps/Config_Files/config_porous2D.txt");
    if (!contfile.is_open()) {
        throw invalid_argument("Config file not found. Expected \"config_porous2D.txt\" in apps/Config_Files.");
    }
    // Default parameters (matching the original code)
    string geometry_file    = "../apps/Geometries/annulus.txt";
    double u_inj            = 0.01;
    double inj_radius_frac  = 0.02;
    double ulb              = 1.0;
    double max_t            = 1000.0;
    double phi_l            = 1.0;
    double phi_g            = 0.0;
    double rho_l            = 1.0;
    double rho_g            = 1e-3;
    double Sigma            = 0.01;
    double xi               = 4.0;
    double M                = 0.1;
    double Fbx              = 0.0;
    double Fby              = 0.0;
    double mu_l             = 0.1;
    double mu_g             = 0.0001;
    double contact_angle_deg = 90.0; // default: natural wetting
    int    natural_wetting   = 1;    // 1 → θ=90°, φ_s=φ_f
    int    N                = 128;
    int    out_freq         = 100;
    int    vtk_freq         = 100;

    // Parse the configuration file
    string line_config, param, value;
    while (getline(contfile, line_config)) {
        // remove comments
        if (auto pos = line_config.find('#'); pos != string::npos) line_config.erase(pos);
        if (line_config.empty()) continue;
        istringstream ls(line_config);
        if (!(ls >> param >> value)) continue;
        if      (param == "inj_vel")         ulb              = stod(value);
        else if (param == "inj_radius")      inj_radius_frac  = stod(value);
        else if (param == "max_t")           max_t            = stod(value);
        else if (param == "out_freq")        out_freq         = stoi(value);
        else if (param == "vtk_freq")        vtk_freq         = stoi(value);
        else if (param == "phi_l")           phi_l            = stod(value);
        else if (param == "phi_g")           phi_g            = stod(value);
        else if (param == "rho_l")           rho_l            = stod(value);
        else if (param == "rho_g")           rho_g            = stod(value);
        else if (param == "Sigma")           Sigma            = stod(value);
        else if (param == "xi")              xi               = stod(value);
        else if (param == "M")               M                = stod(value);
        else if (param == "ulb")             ulb              = stod(value);
        else if (param == "Fbx")             Fbx              = stod(value);
        else if (param == "Fby")             Fby              = stod(value);
        else if (param == "mu_l")            mu_l             = stod(value);
        else if (param == "mu_g")            mu_g             = stod(value);
        else if (param == "N")               N                = stoi(value);
        else if (param == "theta_deg")       contact_angle_deg= stod(value);
        else if (param == "natural_wetting") natural_wetting  = stoi(value);
        else {
            cerr << "Warning: unknown parameter \"" << param << "\"\n";
        }
    }
    contfile.close();

    // --------------------------------------------------------------------
    // 2. Set up the simple geometry (free injection on surface, no file reading)
    int nx = N;
    int ny = N;
    vector<CellType_Porous2D_MRT> flags;
    init_simple_Porous2D_MRT_geometry(flags, nx, ny);

    Dim_Porous2D_MRT dim { nx, ny };
    // lattice‑unit parameters and print them
    auto [Re, At, Ca, Pe, dx, dt] = lbParameters_Porous2D_MRT(ulb, N, mu_l, Sigma, rho_l , rho_g , M);
    printParameters_Porous2D_MRT(dim, Re, At, Ca, Pe, ulb, N, max_t);

    // Allocate distribution arrays for f and g (2*Q populations each)
    vector<CellData> lattice_vect(LBM_Porous2D_MRT::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();

    // Parity flag (for ping–pong memory)
    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];

    // Neighbor tables
    vector<int> nb_vect(dim.nelem * LBM_Porous2D_MRT::Q);
    vector<int> nbb_vect(dim.nelem * LBM_Porous2D_MRT::Q);

    // φ cache
    vector<double> phi_vect(dim.nelem);

    // Wall normals
    vector<double> wall_nx_vect(dim.nelem, 0.0);
    vector<double> wall_ny_vect(dim.nelem, 0.0);

    // Force tracking vectors (optional)
    vector<double> Fs_x_vect(dim.nelem, 0.0);
    vector<double> Fs_y_vect(dim.nelem, 0.0);
    vector<double> Fb_x_vect(dim.nelem, 0.0);
    vector<double> Fb_y_vect(dim.nelem, 0.0);
    vector<double> Fp_x_vect(dim.nelem, 0.0);
    vector<double> Fp_y_vect(dim.nelem, 0.0);

    // D2Q9 constants
    auto [c_vect, opp_vect, w_vect] = d2q9_constants_Porous2D_MRT();

    // Construct the LBM object and assign pointers
    LBM_Porous2D_MRT lbm{};
    lbm.lattice   = lattice;
    lbm.flag      = flags.data();
    lbm.parity    = parity;
    lbm.c         = c_vect.data();
    lbm.opp       = opp_vect.data();
    lbm.w         = w_vect.data();
    lbm.nb        = nb_vect.data();
    lbm.nbb       = nbb_vect.data();
    lbm.wall_nx   = wall_nx_vect.data();
    lbm.wall_ny   = wall_ny_vect.data();
    lbm.phi_cache = phi_vect.data();
    lbm.Fs_x      = Fs_x_vect.data();
    lbm.Fs_y      = Fs_y_vect.data();
    lbm.Fb_x      = Fb_x_vect.data();
    lbm.Fb_y      = Fb_y_vect.data();
    lbm.Fp_x      = Fp_x_vect.data();
    lbm.Fp_y      = Fp_y_vect.data();

    // Assign model parameters
    lbm.phi_l            = phi_l;
    lbm.phi_g            = phi_g;
    lbm.rho_l            = rho_l;
    lbm.rho_g            = rho_g;
    lbm.mu_l             = mu_l;
    lbm.mu_g             = mu_g;
    lbm.Sigma            = Sigma;
    lbm.Fbx              = Fbx;
    lbm.Fby              = Fby;
    lbm.xi               = xi;
    lbm.M                = M;
    lbm.contact_angle_deg= contact_angle_deg;
    lbm.natural_wetting  = (natural_wetting != 0);
    lbm.dim              = dim;
    lbm.u_inj            = u_inj;
    lbm.inj_radius_px    = 0.0;      // set later
    lbm.has_walls        = true;

    // Build neighbor tables
    lbm.build_neighbors();
    // Initialize MRT matrices
    lbm.init_MRT();

    // --------------------------------------------------------------------
    // 3. Determine injection radius and mark injection/outlet cells
    lbm.inj_radius_px = inj_radius_frac * double(min(dim.nx, dim.ny));
    {
        const int   cx = dim.nx/2;
        const int   cy = dim.ny/2;
        const double RAD_SRC_PX = lbm.inj_radius_px;
        const double RAD2       = RAD_SRC_PX * RAD_SRC_PX;
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                const size_t i = lbm.xyz_to_i(x,y);
                if (lbm.flag[i] == CellType_Porous2D_MRT::bounce_back) continue;
                const double dx0 = (x + 0.5) - cx;
                const double dy0 = (y + 0.5) - cy;
                if (dx0*dx0 + dy0*dy0 <= RAD2) {
                    lbm.flag[i] = CellType_Porous2D_MRT::injection;
                } else if (lbm.flag[i] != CellType_Porous2D_MRT::injection) {
                    lbm.flag[i] = CellType_Porous2D_MRT::bulk;
                }
            }
        }
        // Outer boundaries become outlets (unless solid)
        for (int x = 0; x < dim.nx; ++x) {
            size_t iTop = lbm.xyz_to_i(x, dim.ny-1);
            size_t iBot = lbm.xyz_to_i(x, 0);
            if (lbm.flag[iTop] != CellType_Porous2D_MRT::bounce_back) lbm.flag[iTop] = CellType_Porous2D_MRT::outlet;
            if (lbm.flag[iBot] != CellType_Porous2D_MRT::bounce_back) lbm.flag[iBot] = CellType_Porous2D_MRT::outlet;
        }
        for (int y = 0; y < dim.ny; ++y) {
            size_t iL = lbm.xyz_to_i(0, y);
            size_t iR = lbm.xyz_to_i(dim.nx-1, y);
            if (lbm.flag[iL] != CellType_Porous2D_MRT::bounce_back) lbm.flag[iL] = CellType_Porous2D_MRT::outlet;
            if (lbm.flag[iR] != CellType_Porous2D_MRT::bounce_back) lbm.flag[iR] = CellType_Porous2D_MRT::outlet;
        }
    }

    // Close diagonal corners and compute wall normals
    close_solid_corners_Porous2D_MRT(lbm);
    lbm.compute_wall_normals();

    // --------------------------------------------------------------------
    // 4. Initialise distributions
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) {
        lbm.iniLattice(f0);
    });

    // Set up timing and output
    auto [start, clock_iter] = restartClock_Porous2D_MRT();
    ofstream efile("energy.dat");

    // Determine thread count
    int hw = oneapi::tbb::info::default_concurrency();
    int threads = hw;
    if (const char* env = std::getenv("COOLBM_THREADS")) {
        int req = std::atoi(env);
        if (req > 0 && req <= hw) threads = req;
    }
    oneapi::tbb::global_control gc(oneapi::tbb::global_control::max_allowed_parallelism, threads);
    oneapi::tbb::task_arena arena(threads);
    oneapi::tbb::affinity_partitioner ap;

    // Block sizes for parallel_for (can be tuned)
    const int TX = 128;
    const int TY = 64;
    int max_time_iter = static_cast<int>(max_t / dt);

    // --------------------------------------------------------------------
    // 5. Main time‑integration loop
    arena.execute([&]{
        for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
            // Write VTK output periodically
            if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
                lbm.build_phi_cache();          // refresh φ_cache for output
                apply_wetting_bc_phi_Porous2D_MRT(lbm);
                saveVtkFields_Porous2D_MRT(lbm, time_iter);
            }
            // Diagnostic output
            if (out_freq != 0 && time_iter % out_freq == 0) {
                lbm.build_phi_cache();
                apply_wetting_bc_phi_Porous2D_MRT(lbm);
                cout << "Saving profiles at iteration " << time_iter
                     << ", t = " << setprecision(6) << time_iter * dt << setprecision(3)
                     << " [" << (time_iter * dt / max_t * 100.) << "%]" << endl;
                double energy = computeEnergy_Porous2D_MRT(lbm) * dx * dx / (dt * dt);
                cout << "Average energy: " << setprecision(8) << energy << endl;
                efile << setw(12) << setprecision(6) << time_iter * dt
                      << setw(18) << setprecision(8) << energy << "\n";
            }

            // (1) build φ cache
            lbm.build_phi_cache();
            // (2) apply wetting boundary condition on solid walls (φ_s)
            apply_wetting_bc_phi_Porous2D_MRT(lbm);
            // (3) collision–streaming step across all fluid cells
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range2d<int>(0, dim.nx, TX,
                                                  0, dim.ny, TY),
                [&](const oneapi::tbb::blocked_range2d<int>& r){
                    for (int x = r.rows().begin(); x < r.rows().end(); ++x) {
                        size_t base = static_cast<size_t>(x) * static_cast<size_t>(dim.ny);
                        for (int y = r.cols().begin(); y < r.cols().end(); ++y) {
                            int i = static_cast<int>(base + static_cast<size_t>(y));
                            lbm(lbm.lattice[i]);
                        }
                    }
                }, ap
            );
            // (4) injection and outlet boundary conditions
            apply_boundary_conditions_Porous2D_MRT(lbm);
            // swap input and output buffers
            *lbm.parity = 1 - *lbm.parity;
            ++clock_iter;
        }
    });

    // Print final performance
    printMlups_Porous2D_MRT(start, clock_iter, dim.nelem);
}
