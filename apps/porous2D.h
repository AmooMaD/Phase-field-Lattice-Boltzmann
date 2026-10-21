// ───────────────────────── porous2D.h (modified) ──────────────────────────
//
// Phase-field LBM for two-phase flow in porous media, with:
//  - porous geometry read from ASCII .dat file
//  - central circular injection region
//  - open (zero-gradient) external boundaries
//  - geometry-based wall normals
//  - wetting boundary condition à la Fakhari et al. (Eq. 25):
//        natural_wetting = 1  → θ = 90°,  φ_s = φ_f
//        natural_wetting = 0  → general θ via Eq. (25)
//
// --------------------------------------------------------------------------

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

// ───────────────────────── compile-time toggles ─────────────────────────
#ifndef COOLBM_TILE_X
#define COOLBM_TILE_X 128
#endif
#ifndef COOLBM_TILE_Y
#define COOLBM_TILE_Y 64
#endif

// ───────────────────────── cell-level helpers ───────────────────────────
enum class CellType_porous2D : uint8_t { bounce_back = 0, bulk = 1, injection = 2, outlet = 3 };
using  CellData = double;

// ───────────────────────── D2Q9 constants ───────────────────────────────
inline auto d2q9_constants_porous2D()
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

// ───────────────────────── geometry helpers ────────────────────────────
struct Dim_porous2D {
    Dim_porous2D(int nx_, int ny_)
    : nx(nx_), ny(ny_),
    nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)),
    npop (9 * nelem) {}
    int    nx, ny;
    size_t nelem, npop;
};

// ───────────────────────── lattice-unit parameters ─────────────────────
inline auto lbParameters_porous2D(double ulb, int lref, double mu_l, double Sigma, double rho_l, double rho_g, double M)
{
    double Re   = rho_l * ulb * static_cast<double>(lref) / mu_l;
    double At   = (rho_l - rho_g ) / (rho_l + rho_g);
    double Ca   = mu_l * ulb / Sigma;
    double Pe   = ulb * static_cast<double>(lref) / M;
    double dx   = 1. / static_cast<double>(lref);
    double dt   = dx * ulb;
    return make_tuple(Re, At, Ca, Pe, dx, dt);
}

inline void printParameters_porous2D(const Dim_porous2D &dim,
                                     double Re, double At, double Ca, double Pe, double ulb,
                                     int N, double max_t)
{
    cout << "Porous 2-D (Conservative PF — BGK/BGK)\n"
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
inline auto restartClock_porous2D()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_porous2D(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    int dure = static_cast<int>(us.count());
    cout << "Runtime: " << dure/1000000. << " seconds " << endl;
    cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

// ───────────────────────── LBM functor ─────────────────────────────────
// ───────────────────────── LBM functor (Fakhari-like) ─────────────────────────
struct LBM_porous2D
{
    using  CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem + 2 * 9 * nelem; }

    // raw buffers
    CellData*          lattice;
    CellType_porous2D* flag;
    int*               parity;
    std::array<int,2>* c;
    int*               opp;
    double*            t;

    // neighbor tables (precomputed)
    int* nb    = nullptr; // size = nelem*9
    int* nbb   = nullptr; // mirror index (still used by streaming)

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
    Dim_porous2D dim;

    // injection parameters
    double u_inj;
    double inj_radius_px;

    bool has_walls = true;

    static constexpr double cs2 = 1.0/3.0;
    static constexpr double PI  = 3.1415926535897932384626433832795;

    // hydrodynamic MRT data (Fakhari et al. Eq. (27–31))
    bool   use_MRT          = true;      // switch MRT on/off
    bool   use_tau_linear   = false;     // false → Eq. (25), true → Eq. (23) [file:2]
    double s_nu             = 1.0;       // shear relaxation (0 < s_nu < 2)
    double S_diag[9];                    // diagonal entries of S in moment space
    double Mmat[9][9];                   // moment transform M
    double Minv[9][9];                  // M^{-1}

    // ───────── index helpers ─────────
    inline size_t xyz_to_i (int x, int y) const {
        return (size_t)(y + dim.ny * x);
    }
    inline auto i_to_xyz (int i) const {
        int iX = i / (dim.ny);
        int iY = i % (dim.ny);
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
                int i = (int)xyz_to_i(x,y);
                for (int k=0;k<9;++k) {
                    int XX  = (x + c[k][0] + dim.nx) % dim.nx;
                    int YY  = (y + c[k][1] + dim.ny) % dim.ny;
                    int XXb = (x - c[k][0] + dim.nx) % dim.nx;
                    int YYb = (y - c[k][1] + dim.ny) % dim.ny;
                    nb [i*9 + k] = (int)xyz_to_i(XX ,YY );
                    nbb[i*9 + k] = (int)xyz_to_i(XXb,YYb);
                }
            }
        }
    }

    // ───────── MRT initialization (standard D2Q9 moments) ───────
    inline void init_MRT()
    {
        // Lallemand–Luo M matrix [standard choice, same as in many MRT LBMs] [file:2]
        const double Mloc[9][9] = {
            { 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {-4,-1,-1,-1,-1, 2, 2, 2, 2},
            { 4,-2,-2,-2,-2, 1, 1, 1, 1},
            { 0, 1, 0,-1, 0, 1,-1,-1, 1},
            { 0,-2, 0, 2, 0, 1,-1,-1, 1},
            { 0, 0, 1, 0,-1, 1, 1,-1,-1},
            { 0, 0,-2, 0, 2, 1, 1,-1,-1},
            { 0, 1,-1, 1,-1, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 1,-1, 1,-1}
        };
        const double Minv_loc[9][9] = {
            { 1./9, -1./9,  1./9,  0,     0,     0,     0,  0,    0    },
            { 1./9, -1./9,  1./9,  1./6, -1./6,  0,     0,  1./4, 0    },
            { 1./9, -1./9,  1./9,  0,     0,     1./6, -1./6, -1./4, 0 },
            { 1./9, -1./9,  1./9, -1./6,  1./6,  0,     0,  1./4, 0    },
            { 1./9, -1./9,  1./9,  0,     0,    -1./6,  1./6, -1./4, 0 },
            { 1./9,  2./9,  1./9,  1./6,  1./12, 1./6,  1./12, 0,   1./4},
            { 1./9,  2./9,  1./9, -1./6, -1./12, 1./6,  1./12, 0,  -1./4},
            { 1./9,  2./9,  1./9, -1./6,  1./12,-1./6, -1./12, 0,   1./4},
            { 1./9,  2./9,  1./9,  1./6, -1./12,-1./6, -1./12, 0,  -1./4}
        };
        for (int i=0;i<9;++i){
            for (int j=0;j<9;++j){
                Mmat[i][j]   = Mloc[i][j];
                Minv[i][j]   = Minv_loc[i][j];
            }
        }
        // S = diag(1,1,1,1,1,1,1,s_nu,s_nu) as in Eq. (28) [file:2]
        for (int a=0;a<9;++a) S_diag[a] = 1.0;
        S_diag[7] = s_nu;
        S_diag[8] = s_nu;
    }

    // ───────── φ cache, gradients, etc. ───────
    inline void build_phi_cache_tiled(int TX, int TY) {
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range2d<int>(0, dim.nx, TX,
                                              0, dim.ny, TY),
                                  [&](const oneapi::tbb::blocked_range2d<int>& r){
                                      for (int x = r.rows().begin(); x < r.rows().end(); ++x) {
                                          size_t base = (size_t)x * (size_t)dim.ny;
                                          for (int y = r.cols().begin(); y < r.cols().end(); ++y) {
                                              int i = (int)(base + (size_t)y);
                                              if (flag[i] == CellType_porous2D::bounce_back ||
                                                  flag[i] == CellType_porous2D::injection   ||
                                                  flag[i] == CellType_porous2D::outlet) {
                                                  phi_cache[i] = (flag[i] == CellType_porous2D::injection) ? phi_l : phi_g;
                                              continue;
                                                  }
                                                  phi_cache[i] =
                                                  (fin(i,0)+fin(i,2)+fin(i,3)) +
                                                  (fin(i,5)+fin(i,7)+fin(i,8)) +
                                                  (fin(i,6)+fin(i,1)+fin(i,4));
                                          }
                                      }
                                  }
        );
    }
    
    inline std::array<double,2> gradient_phi(int i) const {
        double gx=0.0, gy=0.0;
        const int base = i*9;
        for (int k=0;k<9;++k) {
            int j = nb[base + k];
            double phi_nb = phi_cache[j];
            gx += t[k] * c[k][0] * phi_nb;
            gy += t[k] * c[k][1] * phi_nb;
        }
        gx /= cs2;
        gy /= cs2;
        // Enforce minimum magnitude to prevent division errors
        double gmag = std::sqrt(gx*gx + gy*gy);
        constexpr double gmin = 1e-23;
        if (gmag < gmin) {
            // If gradient is too small, set to minimum with original direction
            if (gmag > 0.0) {
                double scale = gmin / gmag;
                gx *= scale;
                gy *= scale;
            } else {
                gx = gmin;
                gy = 0.0;
            }
        }
        return {gx, gy};
    }
    
    inline double laplacian_phi(int i) const {
        const double phi_c = phi_cache[i];
        double sum = 0.0;
        const int base = i*9;
        for (int k=0;k<9;++k) {
            int j = nb[base + k];
            sum += t[k] * (phi_cache[j] - phi_c);
        }
        return 2. * sum / cs2;
    }

    inline double macro_P_only(int i) const {
        return (gin(i,0)+gin(i,2)+gin(i,3)) +
               (gin(i,5)+gin(i,7)+gin(i,8)) +
               (gin(i,6)+gin(i,1)+gin(i,4));
    }

    inline std::array<double,2> gradient_rho_from_gradphi(const std::array<double,2>& grad_phi) const {
        double delta_rho = (rho_l - rho_g) /(phi_l - phi_g);
        return { delta_rho * grad_phi[0], delta_rho * grad_phi[1] };
    }

    inline double local_rho(double phi) const {
        return rho_g + ((phi - phi_g)/(phi_l - phi_g)) * (rho_l - rho_g);
    }
    inline double local_mu(double phi) const {
        return mu_g + ((phi - phi_g)/(phi_l - phi_g)) * (mu_l - mu_g);
    }

    // Eq. (23) and Eq. (25) options [file:2]
    inline double local_tau_linear(double phi) const {
        double tau_l = mu_l / (rho_l * cs2);
        double tau_g = mu_g / (rho_g * cs2);
        double alpha = (phi - phi_g)/(phi_l - phi_g);
        return tau_g + alpha * (tau_l - tau_g);   // Eq. (23) [file:2]
    }
    inline double local_tau_dyn(double rho, double mu) const {
        // Prevent division by very small rho
        double rho_safe = std::max(rho, 1e-12);
        return mu / (rho_safe * cs2);                  // Eq. (25) [file:2]
    }

    inline double local_nu (double tau) const { return cs2 * (tau); }

    inline double mu_phi(double phi, double lap_phi) const {
        double beta  = 12.0 * Sigma / xi;
        double kappa = 3.0 * Sigma * xi / 2.0;
        double phi0  = (phi_l + phi_g)/2.;
        return 4 * beta * (phi - phi_g) * (phi - phi_l) * (phi - phi0) - kappa * lap_phi;
    }

    inline std::array<double,2> Fs_from(double mu_phi_val, const std::array<double,2>& grad_phi) const {
        return { mu_phi_val * grad_phi[0], mu_phi_val * grad_phi[1] };
    }
    inline std::array<double,2> Fb_from(double rho) const {
        return { rho * Fbx, rho * Fby };
    }
    inline std::array<double,2> Fp_from(double Pstar, double lap_rho) const {
        // Use Laplacian of density (higher-order) for pressure force
        // Physical basis: pressure acts through interfacial curvature, not through density gradients
        // Laplacian is naturally small in bulk (smooth regions) and large only at interfaces
        // This avoids blow-up from large density ratios (rho_g << rho_l)
        // Since we're in 2D and lap_rho is a scalar, return force in direction of gradient
        // For now, apply pressure force proportional to Laplacian magnitude
        return { -Pstar * lap_rho, 0.0 };
    }

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

    inline std::array<double,2> velocity_from(const std::array<double,2>& u_macro,
                                              const std::array<double,2>& F_total,
                                              double rho) const
    {
        // Prevent division by near-zero density
        double rho_safe = std::max(rho, 1e-12);
        return { u_macro[0] + F_total[0]/(2.*rho_safe),
                 u_macro[1] + F_total[1]/(2.*rho_safe) };
    }

    inline std::array<double,2> Fphi_fac_from(double phi, const std::array<double,2>& grad_phi) const {
        double phi0 = (phi_g + phi_l) / 2.;
        double fac  = (1. - 4.*(phi - phi0)*(phi - phi0)) / xi;
        double s    = std::sqrt(grad_phi[0]*grad_phi[0] + grad_phi[1]*grad_phi[1]) + 1e-30;
        return { fac * grad_phi[0] / s, fac * grad_phi[1] / s };
    }

    inline void stream_tab(int i,int k,double pf,double pg)
    {
        if (!has_walls) {
            const int j = nb[i*9 + k];
            fout(j, k) = pf;
            gout(j, k) = pg;
            return;
        }
        const int j = nb[i*9 + k];
        if (flag[j] == CellType_porous2D::bounce_back) {
            fout(i, opp[k]) = pf;
            gout(i, opp[k]) = pg;
        } else {
            fout(j, k) = pf;
            gout(j, k) = pg;
        }
    }

    // ───────── MRT collision on g (velocity-based LB, Eq. 14–21) ───────
    inline void collide_MRT_g(int i,
                              double rho, double Pstar,
                              const std::array<double,2>& u,
                              const std::array<double,2>& F_total,
                              double tau)
    {
        // build g vector
        double gk[9];
        for (int k=0;k<9;++k) gk[k] = gin(i,k);

        // transform to moments: m = M g
        double m[9]; 
        for (int a=0;a<9;++a){
            double s=0.0;
            for (int k=0;k<9;++k) s += Mmat[a][k] * gk[k];
            m[a] = s;
        }

        // equilibrium moments m_eq (pressure-based; simplified Fakhari) [file:2]
        // Here we use standard incompressible D2Q9 equilibrium in moment space.
        double ux = u[0], uy = u[1];
        double usq = ux*ux + uy*uy;

        double m_eq[9];
        m_eq[0] = Pstar;               // density-like
        m_eq[1] = -2.*Pstar + 3.*usq;
        m_eq[2] = Pstar - 3.*usq;
        m_eq[3] = ux;
        m_eq[4] = -ux;
        m_eq[5] = uy;
        m_eq[6] = -uy;
        m_eq[7] = ux*ux - uy*uy;
        m_eq[8] = ux*uy;

        // relaxation
        for (int a=0;a<9;++a) {
            m[a] += -S_diag[a] * (m[a] - m_eq[a]);
        }

        // inverse transform: g* = M^{-1} m
        double g_post[9];
        for (int k=0;k<9;++k){
            double s=0.0;
            for (int a=0;a<9;++a) s += Minv[a][k] * m[a];
            g_post[k] = s;
        }

        // Guo forcing correction in velocity-based LBM (Eq. 15) [file:2]
        double Fx = F_total[0], Fy = F_total[1];
        for (int k=0;k<9;++k) {
            double ck_u = c[k][0]*ux + c[k][1]*uy;
            double Fg   = t[k] * ( (c[k][0]*Fx + c[k][1]*Fy) / cs2 );
            g_post[k]  += (1.0 - 0.5 * (1.0/tau)) * Fg;
        }

        // write back to gin (post-collision, pre-streaming)
        for (int k=0;k<9;++k) gin(i,k) = g_post[k];
    }

    // ───────── BGK collision for f (phase-field) as in your code ───────
    inline std::tuple<double,double,double,double>
    collideBgk_f_and_g(int i,int k,
                       const std::array<double,2>& u,double rho,
                       double phi, double P, double usqr, double tau,
                       const std::array<double,2>& F,
                       const std::array<double,2>& F_phi)
    {
        double ck_u = c[k][0]*u[0] + c[k][1]*u[1];
        double tau_phi = M / cs2;

        double gamma      = t[k] * (1 + ck_u / cs2 + ck_u*ck_u / (2 * cs2 * cs2) - usqr);
        double gamma_opp  = t[k] * (1 - ck_u / cs2 + ck_u*ck_u / (2 * cs2 * cs2) - usqr);
        double geq        = P * t[k] + gamma - t[k];
        double geq_opp    = P * t[k] + gamma_opp - t[k];

        double F_f        = t[k] * (F_phi[0] * c[k][0] + F_phi[1] * c[k][1]);

        double feq_bar      = phi * gamma      - F_f / 2.;
        double feq_bar_opp  = phi * gamma_opp  + F_f / 2.;

        double pf     = fin(i,k)      - (fin(i,k)      - feq_bar     ) / (0.5 + tau_phi) + F_f;
        double pf_opp = fin(i,opp[k]) - (fin(i,opp[k]) - feq_bar_opp ) / (0.5 + tau_phi) - F_f;

        // For g we now let MRT handle collision, so here we just return gin values unchanged
        double pg    = gin(i,k);
        double pg_opp= gin(i,opp[k]);

        return {pf, pg, pf_opp, pg_opp};
    }

    // ───────────────────────── per-cell update ─────────────────────────
    inline void operator()(double& f0)
    {
        int i = (int)(&f0 - lattice);
        if (flag[i] == CellType_porous2D::bounce_back ||
            flag[i] == CellType_porous2D::injection   ||
            flag[i] == CellType_porous2D::outlet) {
            return;
        }

        double phi   = phi_cache[i];
        
        double Pstar = macro_P_only(i);
        double rho   = local_rho(phi);
        double mu    = local_mu(phi);

        double tau   = use_tau_linear ? local_tau_linear(phi) : local_tau_dyn(rho, mu);

        auto  u_macro   = macro_u(i);
        auto  grad_phi  = gradient_phi(i);
        double lap_phi  = laplacian_phi(i);
        auto  grad_rho  = gradient_rho_from_gradphi(grad_phi);
        double lap_rho  = (rho_l - rho_g) * lap_phi;  // Laplacian of density from Laplacian of phi

        double mu_phi_val = mu_phi(phi, lap_phi);
        auto   F_phi      = Fphi_fac_from(phi, grad_phi);

        auto Fs  = Fs_from(mu_phi_val, grad_phi);
        auto Fb  = Fb_from(rho);
        auto Fp  = Fp_from(Pstar, lap_rho);
        
        // Store forces for tracking
        if (Fs_x != nullptr) {
            Fs_x[i] = Fs[0];
            Fs_y[i] = Fs[1];
            Fb_x[i] = Fb[0];
            Fb_y[i] = Fb[1];
            Fp_x[i] = Fp[0];
            Fp_y[i] = Fp[1];
        }

        std::array<double,2> F_total {
            Fs[0] + Fb[0] + Fp[0],
            Fs[1] + Fb[1] + Fp[1]
        };

        auto   u    = velocity_from(u_macro, F_total, rho);
        double usqr = (u[0]*u[0] + u[1]*u[1]) / ( 2. * cs2 );

        // (1) MRT collide hydrodynamics
        if (use_MRT) {
            collide_MRT_g(i, rho, Pstar, u, F_total, tau);
        }

        // (2) BGK for φ (f-lattice) + streaming
        for (int k = 0; k < 4; ++k) {
            auto [pf, pg, pf_op, pg_op] = collideBgk_f_and_g(
                i, k, u, rho, phi, Pstar, usqr, tau, F_total, F_phi
            );
            stream_tab(i, k     , pf   , pg   );
            stream_tab(i, opp[k], pf_op, pg_op);
        }

        // Rest population for f, simple BGK as before
        {
            int k = 4;
            double tau_phi = M / cs2;
            double feq0 = phi * t[k] * (1. - usqr);
            double geq0 = t[k] * (Pstar - usqr);

            fout(i,k) = fin(i,k) - (fin(i,k) - feq0)  / (tau_phi + 0.5);
            gout(i,k) = gin(i,k) - (gin(i,k) - geq0)  / (tau + 0.5);
        }
    }

    // ───────── initialization ─────────
    inline void iniLattice(double& f0)
    {
        int i = int(&f0 - lattice);
        if (flag[i] == CellType_porous2D::bounce_back) {
            for (int k = 0; k < 9; ++k) { fin(i,k)=0.0; gin(i,k)=0.0; }
            return;
        }
        double phi = (flag[i] == CellType_porous2D::injection) ? phi_l : phi_g;
        for (int k = 0; k < 9; ++k) {
            fin(i,k) = phi * t[k];
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

        auto is_fluid = [this, nx, ny, idx](int x, int y) -> bool {
            if (x < 0 || x >= nx || y < 0 || y >= ny) return false;
            int j = idx(x, y);
            return (flag[j] != CellType_porous2D::bounce_back);
        };

        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                int i = idx(x, y);
                if (flag[i] != CellType_porous2D::bounce_back) continue;

                bool   has_fluid = false;
                double nx_acc = 0.0;
                double ny_acc = 0.0;

                // Cardinals
                if (is_fluid(x+1, y)) { nx_acc += 1.0; has_fluid = true; }
                if (is_fluid(x-1, y)) { nx_acc -= 1.0; has_fluid = true; }
                if (is_fluid(x, y+1)) { ny_acc += 1.0; has_fluid = true; }
                if (is_fluid(x, y-1)) { ny_acc -= 1.0; has_fluid = true; }

                // Diagonals if no cardinal
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

    // iniLattice, compute_wall_normals, wetting BC, etc. can stay as you have them.
};

// ───────────────────────── helper: read porous mask ─────────────────────
inline void read_porous_dat(const std::string& fname,
                            std::vector<CellType_porous2D>& flags,
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

    flags.assign((size_t)nx*(size_t)ny, CellType_porous2D::bulk);

    int val;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (!(f >> val)) {
                throw std::runtime_error("Not enough geometry data in " + fname);
            }
            int yy = ny - 1 - y;  // row 0 = top
            size_t idx = (size_t)yy + (size_t)ny * (size_t)x;
            flags[idx] = (val == 1) ? CellType_porous2D::bounce_back
            : CellType_porous2D::bulk;
        }
    }
}

// ───────────────────────── simple geometry: free injection on surface ─────────────────────
inline void init_simple_porous_geometry(std::vector<CellType_porous2D>& flags,
                                        int nx, int ny)
{
    // All cells start as bulk (fluid)
    flags.assign((size_t)nx*(size_t)ny, CellType_porous2D::bulk);
    
    // Optional: add simple solid objects (e.g., a rectangular obstacle in the middle)
    // For now, just leave it all fluid for free injection
    // If you want to add an obstacle, uncomment below:
    /*
    int obs_x1 = nx/3, obs_x2 = 2*nx/3;
    int obs_y1 = ny/3, obs_y2 = 2*ny/3;
    for (int x = obs_x1; x < obs_x2; ++x) {
        for (int y = obs_y1; y < obs_y2; ++y) {
            size_t idx = (size_t)y + (size_t)ny * (size_t)x;
            flags[idx] = CellType_porous2D::bounce_back;
        }
    }
    */
}

// ───────────────────────── close diagonal corners ───────────────────────
inline void close_solid_corners(LBM_porous2D& lbm)
{
    const auto& dim = lbm.dim;

    auto idx = [&](int X,int Y){
        int xw = (X % dim.nx + dim.nx) % dim.nx;      // wrap X
        return (size_t)lbm.xyz_to_i(xw, Y);
    };

    std::vector<uint8_t> add_solid(dim.nelem, 0);

    for (int y = 1; y < dim.ny-1; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x,y);

            if (lbm.flag[i] == CellType_porous2D::bounce_back) continue;

            bool sL = lbm.flag[idx(x-1,y  )] == CellType_porous2D::bounce_back;
            bool sR = lbm.flag[idx(x+1,y  )] == CellType_porous2D::bounce_back;
            bool sD = lbm.flag[idx(x  ,y-1)] == CellType_porous2D::bounce_back;
            bool sU = lbm.flag[idx(x  ,y+1)] == CellType_porous2D::bounce_back;
            bool sUL= lbm.flag[idx(x-1,y+1)] == CellType_porous2D::bounce_back;
            bool sUR= lbm.flag[idx(x+1,y+1)] == CellType_porous2D::bounce_back;
            bool sDL= lbm.flag[idx(x-1,y-1)] == CellType_porous2D::bounce_back;
            bool sDR= lbm.flag[idx(x+1,y-1)] == CellType_porous2D::bounce_back;

            if ((sU && sR && sUR) || (sU && sL && sUL) ||
                (sD && sR && sDR) || (sD && sL && sDL)) {
                add_solid[i] = 1;
                }
        }
    }

    for (size_t i = 0; i < add_solid.size(); ++i) {
        if (!add_solid[i]) continue;
        lbm.flag[i] = CellType_porous2D::bounce_back;
        for (int k = 0; k < 9; ++k) { lbm.fin((int)i,k) = 0.0; lbm.gin((int)i,k) = 0.0; }
    }
}

// ───────────────────────── Injection + outlet BCs (Zou-He for outlets) ────────────────────
inline void apply_boundary_conditions(LBM_porous2D& lbm)
{
    const int nx = lbm.dim.nx;
    const int ny = lbm.dim.ny;
    const double cs2 = LBM_porous2D::cs2;

    // 1) Injection: homogeneous injection with uniform velocity (upward direction)
    // Only set distributions AT injection cells; let streaming naturally propagate them
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int i = (int)lbm.xyz_to_i(x,y);
            if (lbm.flag[i] == CellType_porous2D::injection) {
                // Uniform velocity: upward (positive y direction)
                double ux = 0.0;
                double uy = lbm.u_inj;
                const double u2  = (ux*ux + uy*uy) / (2.0*cs2);
                
                // Set equilibrium ONLY at injection cell
                // Streaming will naturally push this into neighbors
                for (int k = 0; k < 9; ++k) {
                    const double cu    = lbm.c[k][0]*ux + lbm.c[k][1]*uy;
                    const double gamma = lbm.t[k]*(1.0 + cu/cs2 + (cu*cu)/(2.0*cs2*cs2) - u2);
                    lbm.gout(i,k) = gamma;
                    lbm.fout(i,k) = lbm.phi_l * gamma;
                }
            }
        }
    }

    // 2) Zou-He constant pressure outlets
    // Top edge (outlet with P* = 0)
    for (int x = 0; x < nx; ++x) {
        int iT = (int)lbm.xyz_to_i(x, ny-1);
        int jT = (int)lbm.xyz_to_i(x, ny-2);
        if (lbm.flag[iT] == CellType_porous2D::outlet) {
            // Zou-He with constant pressure outlet P* = 0
            double P_outlet = 0.0;
            for (int k = 0; k < 9; ++k) {
                double f_known = lbm.gout(jT, k);
                lbm.gout(iT, k) = f_known - 2.0 * lbm.t[k] * P_outlet * lbm.c[k][1];
            }
            // Phase field at outlet: copy from interior
            for (int k = 0; k < 9; ++k) {
                lbm.fout(iT, k) = lbm.fout(jT, k);
            }
        }
    }
    
    // Bottom edge (outlet)
    for (int x = 0; x < nx; ++x) {
        int iB = (int)lbm.xyz_to_i(x, 0);
        int jB = (int)lbm.xyz_to_i(x, 1);
        if (lbm.flag[iB] == CellType_porous2D::outlet) {
            double P_outlet = 0.0;
            for (int k = 0; k < 9; ++k) {
                double f_known = lbm.gout(jB, k);
                lbm.gout(iB, k) = f_known - 2.0 * lbm.t[k] * P_outlet * lbm.c[k][1];
            }
            for (int k = 0; k < 9; ++k) {
                lbm.fout(iB, k) = lbm.fout(jB, k);
            }
        }
    }
    
    // Left edge (outlet)
    for (int y = 0; y < ny; ++y) {
        int iL = (int)lbm.xyz_to_i(0, y);
        int jL = (int)lbm.xyz_to_i(1, y);
        if (lbm.flag[iL] == CellType_porous2D::outlet) {
            double P_outlet = 0.0;
            for (int k = 0; k < 9; ++k) {
                double f_known = lbm.gout(jL, k);
                lbm.gout(iL, k) = f_known - 2.0 * lbm.t[k] * P_outlet * lbm.c[k][0];
            }
            for (int k = 0; k < 9; ++k) {
                lbm.fout(iL, k) = lbm.fout(jL, k);
            }
        }
    }
    
    // Right edge (outlet)
    for (int y = 0; y < ny; ++y) {
        int iR = (int)lbm.xyz_to_i(nx-1, y);
        int jR = (int)lbm.xyz_to_i(nx-2, y);
        if (lbm.flag[iR] == CellType_porous2D::outlet) {
            double P_outlet = 0.0;
            for (int k = 0; k < 9; ++k) {
                double f_known = lbm.gout(jR, k);
                lbm.gout(iR, k) = f_known - 2.0 * lbm.t[k] * P_outlet * lbm.c[k][0];
            }
            for (int k = 0; k < 9; ++k) {
                lbm.fout(iR, k) = lbm.fout(jR, k);
            }
        }
    }
}

// ───────────────────────── VTK output ────────────────────────────────
inline void saveVtkFields_porous2D(LBM_porous2D& lbm, int time_iter, double dx = 0.) {
    Dim_porous2D const& dim = lbm.dim;
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
            if (lbm.flag[i] == CellType_porous2D::bulk ||
                lbm.flag[i] == CellType_porous2D::outlet ||
                lbm.flag[i] == CellType_porous2D::injection) {
                val = (float)lbm.phi_cache[i];
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
            if (lbm.flag[i] == CellType_porous2D::bulk ||
                lbm.flag[i] == CellType_porous2D::outlet ||
                lbm.flag[i] == CellType_porous2D::injection) {
                val = (float)lbm.local_rho(lbm.phi_cache[i]);
                }
                os << val << " ";
        }
        os << "\n";
    }
    os << "\n";

    // p* * rho / 3
    os << "SCALARS pressure float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double val = 0.;
            if (lbm.flag[i] == CellType_porous2D::bulk ||
                lbm.flag[i] == CellType_porous2D::outlet ||
                lbm.flag[i] == CellType_porous2D::injection) {
                double P   = lbm.macro_P_only(i);
            double rho = lbm.local_rho(lbm.phi_cache[i]);
            val = P * rho / 3.;
                }
                os << val << " ";
        }
        os << "\n";
    }
    os << "\n";

    // Velocity
    os << "VECTORS velocity float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double ux = 0., uy = 0.;
            if (lbm.flag[i] == CellType_porous2D::bulk ||
                lbm.flag[i] == CellType_porous2D::outlet ||
                lbm.flag[i] == CellType_porous2D::injection) {
                double rho    = lbm.local_rho(lbm.phi_cache[i]);
            auto   u_macro  = lbm.macro_u(i);
            auto   grad_phi = lbm.gradient_phi(i);
            double lap_phi  = lbm.laplacian_phi(i);
            double lap_rho  = (lbm.rho_l - lbm.rho_g) * lap_phi;
            double mu_phi_val = lbm.mu_phi(lbm.phi_cache[i], lap_phi);
            auto   Fs     = lbm.Fs_from(mu_phi_val, grad_phi);
            auto   Fb     = lbm.Fb_from(rho);
            double P      = lbm.macro_P_only(i);
            auto   Fp     = lbm.Fp_from(P, lap_rho);
            std::array<double,2> Ftot { Fs[0] + Fb[0] + Fp[0], Fs[1] + Fb[1] + Fp[1] };
            auto u = lbm.velocity_from(u_macro, Ftot, rho);
            ux = u[0]; uy = u[1];
                }
                os << (float)ux << " " << (float)uy << " 0\n";
        }
    }
    os << "\n";

    // Flag
    os << "SCALARS Flag int 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x, y);
            int flag_int = 0;
            if      (lbm.flag[i] == CellType_porous2D::bounce_back) flag_int = 1;
            else if (lbm.flag[i] == CellType_porous2D::injection)   flag_int = 2;
            else if (lbm.flag[i] == CellType_porous2D::outlet)      flag_int = 3;
            os << flag_int << " ";
        }
        os << "\n";
    }
    os << "\n";

    // Wall normals (optional: helpful for debugging)
    os << "VECTORS wall_normal float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            float nxw = 0.f, nyw = 0.f;
            if (lbm.flag[i] == CellType_porous2D::bounce_back) {
                nxw = (float)lbm.wall_nx[i];
                nyw = (float)lbm.wall_ny[i];
            }
            os << nxw << " " << nyw << " 0\n";
        }
    }
    os << "\n";

    // Force tracking (for debugging viscous/pressure/body forces)
    if (lbm.Fs_x != nullptr) {
        os << "VECTORS force_viscous float\n";
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                int i = (int)lbm.xyz_to_i(x, y);
                float fsx = 0.f, fsy = 0.f;
                if (lbm.flag[i] != CellType_porous2D::bounce_back) {
                    fsx = (float)lbm.Fs_x[i];
                    fsy = (float)lbm.Fs_y[i];
                }
                os << fsx << " " << fsy << " 0\n";
            }
        }
        os << "\n";

        os << "VECTORS force_body float\n";
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                int i = (int)lbm.xyz_to_i(x, y);
                float fbx = 0.f, fby = 0.f;
                if (lbm.flag[i] != CellType_porous2D::bounce_back) {
                    fbx = (float)lbm.Fb_x[i];
                    fby = (float)lbm.Fb_y[i];
                }
                os << fbx << " " << fby << " 0\n";
            }
        }
        os << "\n";

        os << "VECTORS force_pressure float\n";
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                int i = (int)lbm.xyz_to_i(x, y);
                float fpx = 0.f, fpy = 0.f;
                if (lbm.flag[i] != CellType_porous2D::bounce_back) {
                    fpx = (float)lbm.Fp_x[i];
                    fpy = (float)lbm.Fp_y[i];
                }
                os << fpx << " " << fpy << " 0\n";
            }
        }
        os << "\n";
    }
}

// ───────────────────────── Energy computation ─────────────────────────
inline double computeEnergy_porous2D(LBM_porous2D& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_porous2D::bulk) {
                auto u_macro  = lbm.macro_u(i);
                std::array<double,2> F0{0,0};
                double rho = lbm.local_rho(lbm.phi_cache[i]);
                auto u = lbm.velocity_from(u_macro, F0, rho);
                energy += u[0]*u[0] + u[1]*u[1];
            }
        }
    }
    return 0.5 * energy;
}

// ───────────────────────── Wetting BC on φ_s (Eq. 25) ──────────────────
// Applies Eq. (25) on all solid wall nodes using wall normals and φ_f.
inline void apply_wetting_bc_phi(LBM_porous2D& lbm)
{
    const auto& dim = lbm.dim;

    // Free-energy parameters
    double beta  = 12.0 * lbm.Sigma / lbm.xi;
    double kappa = 3.0 * lbm.Sigma * lbm.xi / 2.0;
    double factor = std::sqrt(2.0 * beta / kappa); // sqrt(2β/κ)

    double theta_rad  = lbm.contact_angle_deg * LBM_porous2D::PI / 180.0;
    double cos_theta  = std::cos(theta_rad);

    for (int x = 0; x < dim.nx; ++x) {
        for (int y = 0; y < dim.ny; ++y) {
            int i = (int)lbm.xyz_to_i(x,y);
            if (lbm.flag[i] != CellType_porous2D::bounce_back) continue;

            double nxw = lbm.wall_nx[i];
            double nyw = lbm.wall_ny[i];
            if (std::abs(nxw) < 1e-14 && std::abs(nyw) < 1e-14) continue; // not a wall

            // Find fluid neighbor along normal direction
            int base = i * 9;
            int best_k = -1;
            double best_dp = -1e20;
            for (int k = 0; k < 9; ++k) {
                int j = lbm.nb[base + k];
                if (lbm.flag[j] == CellType_porous2D::bounce_back) continue;
                double ex = (double)lbm.c[k][0];
                double ey = (double)lbm.c[k][1];
                double dp = ex*nxw + ey*nyw;
                if (dp > best_dp) {
                    best_dp = dp;
                    best_k = k;
                }
            }
            if (best_k < 0) continue; // no fluid neighbor (should be rare)

            int j = lbm.nb[base + best_k];
            double phi_f = lbm.phi_cache[j];

            double phi_s = phi_f; // default

            if (!lbm.natural_wetting && std::fabs(cos_theta) > 1e-12) {
                // General θ via Eq. (25)
                double ex = (double)lbm.c[best_k][0];
                double ey = (double)lbm.c[best_k][1];
                double dist = std::sqrt(ex*ex + ey*ey); // |x_f - x_s|
                double h = 0.5 * dist;

                double a = -h * factor * cos_theta;

                if (std::fabs(a) < 1e-12) {
                    phi_s = phi_f;
                } else {
                    double term = 1.0 + a;
                    double disc = term*term - 4.0*a*phi_f;
                    if (disc < 0.0) disc = 0.0;
                    double root = std::sqrt(disc);
                    phi_s = (1.0/a)*(1.0 + a - root) - phi_f;
                }
            } else {
                // natural wetting: θ ≈ 90°, φ_s = φ_f
                phi_s = phi_f;
            }

            lbm.phi_cache[i] = phi_s;
        }
    }
}

// ───────────────────────── Geometry init (unused old helper) ───────────
// (kept for completeness; you are using read_porous_dat instead)
template<class LBM>
void iniGeometry(LBM& lbm, double ulb)
{
    std::ifstream gfile("../apps/Geometreis/porous.dat");
    if (!gfile) {
        throw std::invalid_argument("Config file not found. It should be name \"config_combustion2D_geometry.txt\", and placed in the Files_Config directory.");
    }
    if (!gfile.is_open()) {
        throw std::invalid_argument("Can not open geometry configuration file.");
    }

    std::string line;
    while (std::getline(gfile, line)) {
        // no-op
    }
    gfile.close();

    using namespace std;
    cout << "reading Geometry from the file !!" << endl;
    const auto& dim = lbm.dim;

    std::vector<int> flag_file;
    std::ifstream inFil("../data/GeometryFlags.txt");
    if (inFil) {
        double x = 0.0;
        inFil >> x;
        while (inFil) {
            flag_file.push_back(static_cast<int>(x));
            inFil >> x;
        }
        inFil.close();
    } else {
        std::cout << "File could not be opened!!";
    }

    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x, y);
            size_t j = static_cast<size_t>(x) + static_cast<size_t>(dim.nx) * static_cast<size_t>(y);
            if (j < flag_file.size()) {
                int val = flag_file[j];
                if (val == 0) {
                    lbm.flag[i] = CellType_porous2D::bulk;
                } else {
                    lbm.flag[i] = CellType_porous2D::bounce_back;
                }
            } else {
                lbm.flag[i] = CellType_porous2D::bulk;
            }
        }
    }
}

// ───────────────────────── main driver (Porous 2-D) ────────────────
inline void porous2D()
{
    // open configuration file
    ifstream contfile("../apps/Config_Files/config_porous2D.txt");
    if (!contfile.is_open()) {
        throw invalid_argument("Config file not found. Expected \"config_porous2D.txt\" in apps/Config_Files.");
    }

    // Default parameters
    string geometry_file    = "../apps/Geometries/annulus.txt";
    double u_inj            = 0.01;
    double inj_radius_frac  = 0.02;
    double ulb              = 1.0;
    double max_t            = 1000;
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
    int N                  = 128;
    int out_freq           = 100;
    int vtk_freq           = 100;

    string line_config, param, value;
    while (getline(contfile, line_config)) {
        if (auto pos = line_config.find('#'); pos != string::npos) line_config.erase(pos);
        if (line_config.empty()) continue;
        istringstream ls(line_config);
        if (!(ls >> param >> value)) continue;

        else if (param == "inj_vel")         u_inj            = stod(value);
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

    // --------------------------------------------------------------------
    // Setup simple geometry (free injection on surface, no file reading)
    int nx = N;
    int ny = N;
    vector<CellType_porous2D> flags;
    init_simple_porous_geometry(flags, nx, ny);

    Dim_porous2D dim { nx, ny };

    auto [Re, At, Ca, Pe, dx, dt] =
    lbParameters_porous2D(ulb, N, mu_l, Sigma, rho_l , rho_g , M);
    printParameters_porous2D(dim, Re, At, Ca, Pe, ulb, N, max_t);

    // allocate distributions
    vector<CellData> lattice_vect(LBM_porous2D::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();

    // parity
    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];

    // neighbors
    vector<int> nb_vect(dim.nelem*9), nbb_vect(dim.nelem*9);

    // φ cache
    vector<double> phi_vect(dim.nelem);

    // wall normals
    vector<double> wall_nx_vect(dim.nelem, 0.0);
    vector<double> wall_ny_vect(dim.nelem, 0.0);

    // force tracking vectors
    vector<double> Fs_x_vect(dim.nelem, 0.0);
    vector<double> Fs_y_vect(dim.nelem, 0.0);
    vector<double> Fb_x_vect(dim.nelem, 0.0);
    vector<double> Fb_y_vect(dim.nelem, 0.0);
    vector<double> Fp_x_vect(dim.nelem, 0.0);
    vector<double> Fp_y_vect(dim.nelem, 0.0);

    // constants
    auto [c_vect, opp_vect, t_vect] = d2q9_constants_porous2D();
    LBM_porous2D lbm{
        lattice,                // lattice
        flags.data(),           // flag
        parity,                 // parity
        c_vect.data(),          // c
        opp_vect.data(),        // opp
        t_vect.data(),          // t
        /* nb        */ nullptr,
        /* nbb       */ nullptr,
        /* wall_nx   */ nullptr,
        /* wall_ny   */ nullptr,
        /* phi_cache */ nullptr,
        /* Fs_x      */ nullptr,
        /* Fs_y      */ nullptr,
        /* Fb_x      */ nullptr,
        /* Fb_y      */ nullptr,
        /* Fp_x      */ nullptr,
        /* Fp_y      */ nullptr,
        phi_l, phi_g, rho_l, rho_g,
        mu_l, mu_g, Sigma,
        Fbx, Fby, xi, M,
        contact_angle_deg,
        (natural_wetting != 0),
        dim,
        u_inj,
        0.0,       // inj_radius_px (set below)
        true       // has_walls
    };

    lbm.nb        = nb_vect.data();
    lbm.nbb       = nbb_vect.data();
    lbm.phi_cache = phi_vect.data();
    lbm.Fs_x      = Fs_x_vect.data();
    lbm.Fs_y      = Fs_y_vect.data();
    lbm.Fb_x      = Fb_x_vect.data();
    lbm.Fb_y      = Fb_y_vect.data();
    lbm.Fp_x      = Fp_x_vect.data();
    lbm.Fp_y      = Fp_y_vect.data();
    lbm.wall_nx   = wall_nx_vect.data();
    lbm.wall_ny   = wall_ny_vect.data();
    lbm.build_neighbors();

    // Determine injection radius
    lbm.inj_radius_px = inj_radius_frac * double(std::min(dim.nx, dim.ny));

    {
        const int   cx = dim.nx/2;
        const int   cy = dim.ny/2;
        const double RAD_SRC_PX = lbm.inj_radius_px;
        const double RAD2       = RAD_SRC_PX * RAD_SRC_PX;

        // 1) Mark circular injection region
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                const size_t i = lbm.xyz_to_i(x,y);
                if (lbm.flag[i] == CellType_porous2D::bounce_back) continue;

                const double dx0 = (x + 0.5) - cx;
                const double dy0 = (y + 0.5) - cy;
                if (dx0*dx0 + dy0*dy0 <= RAD2) {
                    lbm.flag[i] = CellType_porous2D::injection;
                } else if (lbm.flag[i] != CellType_porous2D::injection) {
                    lbm.flag[i] = CellType_porous2D::bulk;
                }
            }
        }

        // 2) Make outer boundaries outlets (if not solid)
        for (int x = 0; x < dim.nx; ++x) {
            size_t iTop = lbm.xyz_to_i(x, dim.ny-1);
            size_t iBot = lbm.xyz_to_i(x, 0);
            if (lbm.flag[iTop] != CellType_porous2D::bounce_back) lbm.flag[iTop] = CellType_porous2D::outlet;
            if (lbm.flag[iBot] != CellType_porous2D::bounce_back) lbm.flag[iBot] = CellType_porous2D::outlet;
        }
        for (int y = 0; y < dim.ny; ++y) {
            size_t iL = lbm.xyz_to_i(0, y);
            size_t iR = lbm.xyz_to_i(dim.nx-1, y);
            if (lbm.flag[iL] != CellType_porous2D::bounce_back) lbm.flag[iL] = CellType_porous2D::outlet;
            if (lbm.flag[iR] != CellType_porous2D::bounce_back) lbm.flag[iR] = CellType_porous2D::outlet;
        }
    }

    close_solid_corners(lbm);
    lbm.compute_wall_normals();

    // Initialize distributions
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });

    auto [start, clock_iter] = restartClock_porous2D();
    ofstream efile("energy.dat");

    int hw = oneapi::tbb::info::default_concurrency();
    int threads = hw;
    if (const char* env = std::getenv("COOLBM_THREADS")) {
        int req = std::atoi(env);
        if (req > 0 && req <= hw) threads = req;
    }
    oneapi::tbb::global_control gc(oneapi::tbb::global_control::max_allowed_parallelism, threads);
    oneapi::tbb::task_arena arena(threads);
    oneapi::tbb::affinity_partitioner ap;

    const int TX = 128;
    const int TY = 64;

    int max_time_iter = static_cast<int>(max_t / dt);

    arena.execute([&]{
        for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {

            if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
                lbm.build_phi_cache_tiled(TX, TY);      // keep φ_cache fresh for output
                apply_wetting_bc_phi(lbm);              // update φ_s before writing
                saveVtkFields_porous2D(lbm, time_iter);
            }
            if (out_freq != 0 && time_iter % out_freq == 0) {
                lbm.build_phi_cache_tiled(TX, TY);
                apply_wetting_bc_phi(lbm);
                cout << "Saving profiles at iteration " << time_iter
                << ", t = " << setprecision(6) << time_iter * dt << setprecision(3)
                << " [" << (time_iter * dt / max_t * 100.) << "%]" << endl;
                double energy = computeEnergy_porous2D(lbm) * dx * dx / (dt * dt);
                cout << "Average energy: " << setprecision(8) << energy << endl;
                efile << setw(12) << setprecision(6) << time_iter * dt
                << setw(18) << setprecision(8) << energy << "\n";
            }

            // 1) build φ cache
            lbm.build_phi_cache_tiled(TX, TY);
            // 2) apply wetting BC on walls: φ_s(θ)
            apply_wetting_bc_phi(lbm);

            // 3) collide+stream
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range2d<int>(0, dim.nx, TX,
                                                  0, dim.ny, TY),
                                      [&](const oneapi::tbb::blocked_range2d<int>& r){
                                          for (int x = r.rows().begin(); x < r.rows().end(); ++x) {
                                              size_t base = (size_t)x * (size_t)dim.ny;
                                              for (int y = r.cols().begin(); y < r.cols().end(); ++y) {
                                                  int i = (int)(base + (size_t)y);
                                                  lbm(lbm.lattice[i]);
                                              }
                                          }
                                      },
                                      ap
            );

            // 4) injection + outlets
            apply_boundary_conditions(lbm);

            *lbm.parity = 1 - *lbm.parity;
            ++clock_iter;
        }
    });

    printMlups_porous2D(start, clock_iter, dim.nelem);
}
