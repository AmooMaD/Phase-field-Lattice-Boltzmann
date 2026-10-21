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
#include <limits>
#include <random>
#include <cstdint>


// oneTBB
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range3d.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/partitioner.h>

using namespace std;
using namespace std::chrono;

// ─────────────────────────   compile-time toggles   ────────────────────────
#ifndef COOLBM_TILE_X
#define COOLBM_TILE_X 128
#endif
#ifndef COOLBM_TILE_Y
#define COOLBM_TILE_Y 128
#endif
#ifndef COOLBM_TILE_Z
#define COOLBM_TILE_Z 128
#endif
// #define COOLBM_NAN_GUARD 1  // enable only for debugging

// ─────────────────────────   cell-level helpers   ───────────────────────────
enum class CellType_porous3D_Fakhari : uint8_t { bounce_back, bulk };
using  CellData = double;

// ─────────────────────────   D3Q19 constants   ───────────────────────────────
inline auto d3q19_constants_porous3D_Fakhari()
{
    vector<array<int, 3>> c_vect = {
        {-1, 0, 0}, { 0,-1, 0}, { 0, 0,-1},
        {-1,-1, 0}, {-1, 1, 0}, {-1, 0,-1},
        {-1, 0, 1}, { 0,-1,-1}, { 0,-1, 1},
        { 0, 0, 0},
        { 1, 0, 0}, { 0, 1, 0}, { 0, 0, 1},
        { 1, 1, 0}, { 1,-1, 0}, { 1, 0, 1},
        { 1, 0,-1}, { 0, 1, 1}, { 0, 1,-1}
    };

    // The opposite of a given direction.
    vector<int> opp_vect =
    { 10, 11, 12, 13, 14, 15, 16, 17, 18, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8 };

    // The lattice weights.
    vector<double> t_vect =
    {
        1./18., 1./18., 1./18., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36.,
        1./3.,
        1./18., 1./18., 1./18., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36.
    };
    return make_tuple(c_vect, opp_vect, t_vect);
}

// ─────────────────────────   geometry helpers   ────────────────────────────
struct Dim_porous3D_Fakhari {

    Dim_porous3D_Fakhari(int nx_, int ny_, int nz_)
    : nx(nx_), ny(ny_), nz(nz_),
    nelem(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz)),
    npop(19 * nelem)
    { }
    int nx, ny, nz;
    size_t nelem, npop;
};

// ─────────────────────────   lattice-unit parameters   ─────────────────────
inline auto lbParameters_porous3D_Fakhari(double ulb, int lref, double mu_l, double Sigma, double rho_l, double rho_g, double M)
{
    double Re   = rho_l * ulb * static_cast<double>(lref) / mu_l;
    double At   = (rho_l - rho_g ) / (rho_l + rho_g);
    double Ca   = mu_l * ulb / Sigma;
    double Pe   = ulb * static_cast<double>(lref) / M;
    double dx   = 1. / static_cast<double>(lref);
    double dt   = dx * ulb;
    return make_tuple(Re, At, Ca, Pe, dx, dt);
}

inline void printParameters_porous3D_Fakhari(const Dim_porous3D_Fakhari &dim,
                                                     double Re, double At, double Ca, double Pe, double ulb,
                                                     int N, double max_t)
{
    cout << "Porous meida 3-D (Conservative PF — BGK/BGK)\n"
    << "N      = " << N        << '\n'
    << "nx     = " << dim.nx   << '\n'
    << "ny     = " << dim.ny   << '\n'
    << "nz     = " << dim.nz   << '\n'
    << "Re     = " << Re       << '\n'
    << "At     = " << At       << '\n'
    << "Ca     = " << Ca       << '\n'
    << "Pe     = " << Pe       << '\n'
    << "ulb    = " << ulb      << '\n'
    << "max_t  = " << max_t    << '\n';
}

// ─────────────────────────   timing helpers   ──────────────────────────────
inline auto restartClock_porous3D_Fakhari()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_porous3D_Fakhari(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    int dure = static_cast<int>(us.count());
    cout << "Runtime: " << dure/1000000. << " seconds " << endl;
    cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

// ─────────────────────────   LBM functor   ─────────────────────────────────
struct LBM_porous3D_Fakhari
{
    using  CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 19 * nelem + 2 * 19 * nelem; }

    // raw buffers
    CellData*                         lattice;
    CellType_porous3D_Fakhari* flag;
    int*                              parity;
    std::array<int,3>*                c;
    int*                              opp;
    double*                           t;

    // neighbor tables (precomputed)
    int* nb    = nullptr; // size = nelem*9
    int* nbb   = nullptr; // mirror index for bounce-back along k (size = nelem*9)

    // cached φ (scalar) per cell per step
    double* phi_cache = nullptr; // size = nelem

    // model params
    double phi_l;
    double phi_g;       // 1 and 0 (or ±0.5)
    double rho_l;
    double rho_g;       // heavy (l), light (g)
    double mu_l;        // viscosity of liquid
    double mu_g;        // viscosity of gas
    double Sigma;
    double Fbx;         // Body force in x
    double Fby;         // Body force in y
    double Fbz;
    double xi;
    double M;
    Dim_porous3D_Fakhari dim;

    // topology fast-path
    bool has_walls = true;

    // constants
    static constexpr double cs2 = 1.0/3.0;
    static constexpr double PI  = 3.1415926535897932384626433832795;

    // ───────── index helpers ─────────
    auto i_to_xyz (int i) {
        int iX = i / (dim.ny * dim.nz);
        int remainder = i % (dim.ny * dim.nz);
        int iY = remainder / dim.nz;
        int iZ = remainder % dim.nz;
        return std::make_tuple(iX, iY, iZ);
    };

    // Convert Cartesian indices to linear index.
    size_t xyz_to_i (int x, int y, int z) {
        return z + dim.nz * (y + dim.ny * x);
    };


    // ───────── array accessors (ping-pong) ───────
    // non-const (write paths)
    inline double& f   (int i,int k)        { return lattice[              k*dim.nelem + i]; }
    inline double& fin (int i,int k)        { return lattice[*parity  *dim.npop + k*dim.nelem + i]; }
    inline double& fout(int i,int k)        { return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }
    inline double& g   (int i,int k)        { return lattice[2*dim.npop                  + k*dim.nelem + i]; }
    inline double& gin (int i,int k)        { return lattice[*parity  *dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    inline double& gout(int i,int k)        { return lattice[(1-*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    // const (read paths in const methods)
    inline double  f   (int i,int k)  const { return lattice[              k*dim.nelem + i]; }
    inline double  fin (int i,int k)  const { return lattice[*parity  *dim.npop + k*dim.nelem + i]; }
    inline double  fout(int i,int k)  const { return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }
    inline double  g   (int i,int k)  const { return lattice[2*dim.npop                  + k*dim.nelem + i]; }
    inline double  gin (int i,int k)  const { return lattice[*parity  *dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    inline double  gout(int i,int k)  const { return lattice[(1-*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    // ───────── neighbor table builder (call once after lbm is constructed) ───────
    inline void build_neighbors()
    {
        for (int x = 0; x < dim.nx; ++x) {
            for (int y = 0; y < dim.ny; ++y) {
                for (int z = 0; z < dim.nz; ++z) {
                    int i = (int)xyz_to_i(x,y,z);
                    for (int k=0;k<19;++k) {
                        int XX  = (x + c[k][0] + dim.nx) % dim.nx;
                        int YY  = (y + c[k][1] + dim.ny) % dim.ny;
                        int ZZ  = (z + c[k][2] + dim.nz) % dim.nz;
                        //int YY  = y + c[k][1];
                        int XXb = (x - c[k][0] + dim.nx) % dim.nx;
                        int YYb = (y - c[k][1] + dim.ny) % dim.ny;
                        int ZZb = (z - c[k][2] + dim.nz) % dim.nz;
                        //int YYb = y - c[k][1];
                        nb [i*19 + k] = (int)xyz_to_i(XX ,YY ,ZZ);
                        nbb[i*19 + k] = (int)xyz_to_i(XXb,YYb,ZZb);
                    }
                }
            }
        }
    }


    // ───────── build φ cache (once per step, tiled) ─────────
    inline void build_phi_cache_tiled(int TX, int TY, int TZ) {
        using oneapi::tbb::blocked_range3d;
        using oneapi::tbb::parallel_for;

        parallel_for(
            blocked_range3d<int>(
                0, dim.nz, TZ,   // pages (z)
                0, dim.ny, TY,   // rows  (y)
                0, dim.nx, TX    // cols  (x)
            ),
            [&](const blocked_range3d<int>& r) {
                for (int z = r.pages().begin(); z < r.pages().end(); ++z) {
                    size_t zbase = (size_t)z * (size_t)dim.nx * (size_t)dim.ny;
                    for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
                        size_t ybase = zbase + (size_t)y * (size_t)dim.nx;
                        for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                            const int i = (int)xyz_to_i(x,y,z);  // or: int i = xyz_to_i(x,y,z);

                                // Sum all D3Q19 populations (order doesn’t matter for the sum):
                                double s =
                                fin(i, 0)  + fin(i, 1)  + fin(i, 2)  + fin(i, 3)  + fin(i, 4)
                                + fin(i, 5)  + fin(i, 6)  + fin(i, 7)  + fin(i, 8)  + fin(i, 9)
                                + fin(i,10)  + fin(i,11) + fin(i,12) + fin(i,13) + fin(i,14)
                                + fin(i,15) + fin(i,16) + fin(i,17) + fin(i,18);

                                phi_cache[i] = s;
                        }
                    }
                }
            }
        );
    }


    // ───────────────────────── initialization (droplet; gin=0) ──────────────────────────────
    inline void iniLattice(double& f0)
    {
        const int i = int(&f0 - lattice);

        if (flag[i] == CellType_porous3D_Fakhari::bounce_back) {
            for (int k = 0; k < 19; ++k) { fin(i,k)=0.0; gin(i,k)=0.0; }
            return;
        }

        auto [iX, iY, iZ] = i_to_xyz(i);
        (void)iY; (void)iZ;                 // 2D case uses x only
        const double X    = double(iX) + 0.5;
        const double nx_d = double(dim.nx);

        // Heavy on both sides, light in the middle:
        // interfaces at xL (heavy→light) and xR (light→heavy)
        const double xL = 0.10 * nx_d;      // 10% from left
        const double xR = 0.90 * nx_d;      // 10% from right
        const double w  = 0.5 * xi;         // smoothing half-width; thickness ~ 4w

        // Helper: tanh blend between phi_l and phi_g centered at x0
        auto transition = [&](double x0, bool heavy_to_light) {
            const double s    = std::tanh((X - x0)/w);
            const double phi0 = 0.5*(phi_l + phi_g);
            const double amp  = 0.5*(phi_l - phi_g);
            // heavy_to_light: X<<x0 ⇒ ~phi_l, X>>x0 ⇒ ~phi_g
            return heavy_to_light ? (phi0 - amp*s) : (phi0 + amp*s);
        };

        // Left profile: heavy→light at xL
        const double phi_L = transition(xL, /*heavy_to_light=*/true);
        // Right profile: light→heavy at xR
        const double phi_R = transition(xR, /*heavy_to_light=*/false);

        // Smoothly blend left and right profiles around the mid region
        const double mid = 0.5*(xL + xR);
        const double wM  = w;                               // blending width; can be wider if you like
        const double alpha = 0.5*(1.0 + std::tanh((X - mid)/wM)); // 0 (left) → 1 (right)

        const double phi = (1.0 - alpha)*phi_L + alpha*phi_R;     // smooth, C^1 transition

        // Initialize PDFs at rest (u=0 ⇒ weights t[k])
        for (int k = 0; k < 19; ++k) {
            fin(i,k) = phi * t[k];
            gin(i,k) = 0.0;
        }
    }



    // ───────────────────────── kernels ─────────────────────────

    inline auto macro_phi_P(int i) const {
        double phi=0., P=0.;
        for (int k=0;k<19;++k){ phi+=fin(i,k); P+=gin(i,k); }
        return std::make_tuple(phi,P);
    }
    inline double macro_P_only(int i) const {
        double P=0.; for (int k=0;k<19;++k) P+=gin(i,k); return P;
    }


    inline std::array<double,3> macro_u(int i) const {
        double ux=0., uy=0., uz=0.;
        for (int k=0;k<19;++k){ ux += gin(i,k)*c[k][0]; uy += gin(i,k)*c[k][1]; uz += gin(i,k)*c[k][2]; }
        return {ux, uy, uz};
    }


    inline double local_rho(double phi) const {
        return rho_g + ((phi - phi_g)/(phi_l - phi_g)) * (rho_l - rho_g);
    }
    inline double local_mu(double phi) const {
        return mu_g + ((phi - phi_g)/(phi_l - phi_g)) * (mu_l - mu_g);
    }
    inline double local_tau(double rho, double mu) const { return mu / (rho * cs2); }
    inline double local_nu (double tau) const { return tau * cs2; }

    inline std::array<double,3> gradient_phi(int i) const {
        double gx=0., gy=0., gz = 0.;
        const int base = i*19;
        for (int k=0;k<19;++k) {
            int j = nb[base + k];
            if (has_walls && flag[j] == CellType_porous3D_Fakhari::bounce_back)
                j = nbb[base + k];
            double phi_nb = phi_cache[j]; // cached scalar
            gx += t[k] * c[k][0] * phi_nb;
            gy += t[k] * c[k][1] * phi_nb;
            gz += t[k] * c[k][2] * phi_nb;
        }
        return { gx / cs2, gy / cs2, gz / cs2};
    }

    inline double laplacian_phi(int i) const {
        const double phi_c = phi_cache[i];
        double sum = 0.0;
        const int base = i*19;
        for (int k=0;k<19;++k) {
            int j = nb[base + k];
            if (has_walls && flag[j] == CellType_porous3D_Fakhari::bounce_back)
                j = nbb[base + k];  // mirror across the wall
                sum += t[k] * (phi_cache[j] - phi_c);
        }
        return 2. * sum / cs2;
    }


    inline std::array<double,3> gradient_rho_from_gradphi(const std::array<double,3>& grad_phi) const {
        double delta_rho = (rho_l - rho_g) /(phi_l - phi_g);
        return { delta_rho * grad_phi[0], delta_rho * grad_phi[1], delta_rho * grad_phi[2]};
    }

    inline double mu_phi(double phi, double lap_phi) const {
        double beta  = 12.0 * Sigma / xi;
        double kappa = 3. * Sigma * xi / 2.;
        double phi0  = (phi_l + phi_g)/2.;
        return 4 * beta * (phi - phi_g) * (phi - phi_l) * (phi - phi0) - kappa * lap_phi;
    }

    inline std::array<double,3> Fs_from(double mu_phi_val, const std::array<double,3>& grad_phi) const {
        return { mu_phi_val * grad_phi[0], mu_phi_val * grad_phi[1], mu_phi_val * grad_phi[2] };
    }
    inline std::array<double,3> Fb_from(double rho) const {
        return { rho * Fbx, rho * Fby, rho * Fbz };
    }
    inline std::array<double,3> Fp_from(double Pstar, const std::array<double,3>& grad_rho) const {
        return { -Pstar * cs2 * grad_rho[0], -Pstar * cs2 * grad_rho[1], -Pstar * cs2 * grad_rho[2] };
    }

    inline std::array<double,3> Fmu_from(int i,
                                         const std::array<double,3>& u,
                                         double pstar, double tau, double nu,
                                         const std::array<double,3>& grad_rho) const
                                         {
                                             double usqr = (u[0]*u[0] + u[1]*u[1]+ u[2]*u[2]) / (2.0*cs2);
                                             double fac  = - nu / ((tau + 0.5) * cs2);

                                             double Sxx = 0.0, Sxy = 0.0, Syy = 0.0, Sxz = 0., Syz = 0. , Szz = 0.;
                                             for (int k=0; k<19; ++k) {
                                                 if (k==9) continue;
                                                 double ex = (double)c[k][0];
                                                 double ey = (double)c[k][1];
                                                 double ez = (double)c[k][2];
                                                 double ck_u = ex*u[0] + ey*u[1] + ez*u[2];
                                                 double gamma_k = t[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - usqr);
                                                 double geq_k   = pstar * t[k] + (gamma_k - t[k]);
                                                 double gneq    = gin(i,k) - geq_k;
                                                 Sxx += gneq * ex * ex;
                                                 Sxy += gneq * ex * ey;
                                                 Syy += gneq * ey * ey;
                                                 Sxz += gneq * ex * ez;
                                                 Syz += gneq * ey * ez;
                                                 Szz += gneq * ez * ez;
                                             }

                                             double Fx = fac * ( Sxx * grad_rho[0] + Sxy * grad_rho[1] + Sxz * grad_rho[2] );
                                             double Fy = fac * ( Sxy * grad_rho[0] + Syy * grad_rho[1] + Syz * grad_rho[2] );
                                             double Fz = fac * ( Sxz * grad_rho[0] + Syz * grad_rho[1] + Szz * grad_rho[2] );
                                             return {Fx, Fy, Fz};
                                         }

                                         inline std::array<double,3> velocity_from(const std::array<double,3>& u_macro,
                                                                                   const std::array<double,3>& F_total,
                                                                                   double rho) const
                                                                                   {
                                                                                       return { u_macro[0] + F_total[0]/(2.*rho),
                                                                                                u_macro[1] + F_total[1]/(2.*rho),
                                                                                                u_macro[2] + F_total[2]/(2.*rho) };
                                                                                   }

                                                                                   inline std::array<double,3> Fphi_fac_from(double phi, const std::array<double,3>& grad_phi) const {
                                                                                       double phi0 = (phi_g + phi_l) / 2.;
                                                                                       double fac  = (1 - 4*(phi - phi0)*(phi - phi0)) / xi;
                                                                                       double s    = std::sqrt(grad_phi[0]*grad_phi[0] + grad_phi[1]*grad_phi[1] + grad_phi[2]*grad_phi[2] ) + 1e-30;
                                                                                       return { fac * grad_phi[0] / s, fac * grad_phi[1] / s, fac * grad_phi[2] / s };
                                                                                   }

                                                                                   inline void stream_tab(int i,int k,double pf,double pg)
                                                                                   {
                                                                                       if (!has_walls) {
                                                                                           const int j = nb[i*19 + k];
                                                                                           fout(j, k) = pf;
                                                                                           gout(j, k) = pg;
                                                                                           return;
                                                                                       }
                                                                                       const int j = nb[i*19 + k];
                                                                                       if (flag[j] == CellType_porous3D_Fakhari::bounce_back) {
                                                                                           fout(i, opp[k]) = pf;
                                                                                           gout(i, opp[k]) = pg;
                                                                                       } else {
                                                                                           fout(j, k) = pf;
                                                                                           gout(j, k) = pg;
                                                                                       }
                                                                                   }

                                                                                   inline std::tuple<double,double,double,double>
                                                                                   collideBgk(int i,int k,
                                                                                              const std::array<double,3>& u,double rho,
                                                                                              double phi, double P, double usqr, double tau,
                                                                                              const std::array<double,3>& F,
                                                                                              const std::array<double,3>& F_phi)
                                                                                   {
                                                                                       double ck_u = c[k][0]*u[0] + c[k][1]*u[1] + c[k][2]*u[2];
                                                                                       double tau_phi = M / cs2;

                                                                                       double gamma      = t[k] * (1 + ck_u / cs2 + ck_u*ck_u / (2 * cs2 * cs2) - usqr);
                                                                                       double gamma_opp  = t[k] * (1 - ck_u / cs2 + ck_u*ck_u / (2 * cs2 * cs2) - usqr);
                                                                                       double geq        = P * t[k] + gamma - t[k];
                                                                                       double geq_opp    = P * t[k] + gamma_opp - t[k];

                                                                                       double F_f        = t[k] * (F_phi[0] * c[k][0] + F_phi[1] * c[k][1] + F_phi[2] * c[k][2]);

                                                                                       double feq_bar      = phi * gamma      - F_f / 2.;
                                                                                       double feq_bar_opp  = phi * gamma_opp  + F_f / 2.;

                                                                                       double pf     = fin(i,k)      - (fin(i,k)      - feq_bar     ) / (0.5 + tau_phi) + F_f;
                                                                                       double pf_opp = fin(i,opp[k]) - (fin(i,opp[k]) - feq_bar_opp ) / (0.5 + tau_phi) - F_f;

                                                                                       double F_g    = /*(1 - 0.5/tau)* */ t[k] * (F[0]*c[k][0] + F[1]*c[k][1] + F[2] * c[k][2]) / (rho * cs2);

                                                                                       double geq_bar     = geq     - F_g/2.;
                                                                                       double geq_bar_opp = geq_opp + F_g/2.;

                                                                                       double pg     = gin(i,k)      - (gin(i,k)      - geq_bar    ) / (0.5 + tau) + F_g;
                                                                                       double pg_opp = gin(i,opp[k]) - (gin(i,opp[k]) - geq_bar_opp) / (0.5 + tau) - F_g;

                                                                                       return {pf, pg, pf_opp, pg_opp};
                                                                                   }

                                                                                   inline void operator()(double& f0)
                                                                                   {
                                                                                       int i = (int)(&f0 - lattice);
                                                                                       if (flag[i] != CellType_porous3D_Fakhari::bulk) return;

                                                                                       // --- use cached φ; sum P* only
                                                                                       double phi   = phi_cache[i];
                                                                                       double Pstar = macro_P_only(i);
                                                                                       double rho  = local_rho(phi);
                                                                                       double mu   = local_mu(phi);
                                                                                       double tau  = local_tau(rho, mu);
                                                                                       double nu   = local_nu(tau);

                                                                                       #ifdef COOLBM_NAN_GUARD
                                                                                       auto ok = [](double v){ return std::isfinite(v); };
                                                                                       if (!ok(phi)||!ok(Pstar)||!ok(rho)||!ok(mu)||!ok(tau)||!ok(nu)) {
                                                                                           throw std::runtime_error("Non-finite macro detected.");
                                                                                       }
                                                                                       #endif

                                                                                       auto u_macro = macro_u(i);
                                                                                       auto grad_phi = gradient_phi(i);
                                                                                       double lap_phi = laplacian_phi(i);
                                                                                       auto grad_rho  = gradient_rho_from_gradphi(grad_phi);

                                                                                       double mu_phi_val = mu_phi(phi, lap_phi);
                                                                                       auto F_phi = Fphi_fac_from(phi, grad_phi);

                                                                                       auto Fs  = Fs_from(mu_phi_val, grad_phi);
                                                                                       auto Fb  = Fb_from(rho);
                                                                                       auto Fp  = Fp_from(Pstar, grad_rho);
                                                                                       auto Fmu = Fmu_from(i, u_macro, Pstar, tau, nu, grad_rho);

                                                                                       std::array<double,3> F_total { Fs[0] + Fb[0] + Fp[0] + Fmu[0],
                                                                                                                      Fs[1] + Fb[1] + Fp[1] + Fmu[1],
                                                                                                                      Fs[2] + Fb[2] + Fp[2] + Fmu[2] };

                                                                                           auto u = velocity_from(u_macro, F_total, rho);
                                                                                           double usqr = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) / ( 2. * cs2 );

                                                                                           for (int k = 0; k < 9; ++k) {
                                                                                               auto [pf, pg, pf_op, pg_op] = collideBgk(
                                                                                                   i, k, u, rho, phi, Pstar, usqr, tau, F_total, F_phi
                                                                                               );
                                                                                               stream_tab(i, k     , pf   , pg   );
                                                                                               stream_tab(i, opp[k], pf_op, pg_op);
                                                                                           }

                                                                                           // Rest population (k = 4)
                                                                                           {
                                                                                               int k = 9;
                                                                                               double tau_phi = M / cs2;

                                                                                               double feq0 = phi * t[k] * (1. - usqr);
                                                                                               double geq0 = t[k] * (Pstar - usqr);

                                                                                               double F_g  = 0.0;
                                                                                               double geq_bar0 = geq0 - 0.5 * F_g;

                                                                                               fout(i,k) = fin(i,k) - (fin(i,k) - feq0) / (tau_phi + 0.5);
                                                                                               gout(i,k) = gin(i,k) - (gin(i,k) - geq_bar0) / (tau + 0.5) + F_g;
                                                                                           }
                                                                                   }





};


// ───────────────────────── VTK output ─────────────────────────────────
inline void saveVtkFields_porous3D_Fakhari(LBM_porous3D_Fakhari& lbm, int time_iter, double dx = 0.)
{
    Dim_porous3D_Fakhari const& dim = lbm.dim;
    if (dx == 0.0) dx = 1.0 / static_cast<double>(dim.nx);

    std::stringstream ss;
    ss << "sol_" << std::setw(7) << std::setfill('0') << time_iter << ".vtk";
    std::ofstream os(ss.str());
    os << "# vtk DataFile Version 2.0\n";
    os << "iteration " << time_iter << "\n";
    os << "ASCII\n\n";
    os << "DATASET STRUCTURED_POINTS\n";
    os << "DIMENSIONS " << dim.nx << " " << dim.ny << " " << dim.nz << "\n";   // nz should be 1 for 2D
    os << "ORIGIN 0 0 0\n";
    os << "SPACING " << dx << " " << dx << " " << dx << "\n\n";
    os << "POINT_DATA " << (dim.nx * dim.ny * dim.nz) << "\n";

    // Helper: get linear index. For true 3D storage, replace with xyz_to_i(x,y,z).
    auto idx3D = [&](int x, int y, int z) -> int {
        return static_cast<int>(lbm.xyz_to_i(x, y, z)); // if you have xyz_to_i(x,y,z), use that instead
    };

    // ------------------ φ ------------------
    os << "SCALARS phi float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int z = 0; z < dim.nz; ++z) {            // 2D: runs once when nz==1
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                const int i = idx3D(x, y, z);        // or: lbm.xyz_to_i(x,y,z)
                float val = 0.f;
                if (lbm.flag[i] == CellType_porous3D_Fakhari::bulk) {
                    val = static_cast<float>(lbm.phi_cache[i]); // assumed built each step
                }
                os << val << " ";
            }
            os << "\n";
        }
    }
    os << "\n";

    // ------------------ ρ ------------------
    os << "SCALARS density float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int z = 0; z < dim.nz; ++z) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                const int i = idx3D(x, y, z);        // or: lbm.xyz_to_i(x,y,z)
                float val = 0.f;
                if (lbm.flag[i] == CellType_porous3D_Fakhari::bulk) { // fixed wrong enum
                    val = static_cast<float>(lbm.local_rho(lbm.phi_cache[i]));
                }
                os << val << " ";
            }
            os << "\n";
        }
    }
    os << "\n";

    // -------------- p* * rho / 3 --------------
    os << "SCALARS pressure float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int z = 0; z < dim.nz; ++z) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                const int i = idx3D(x, y, z);
                double val = 0.0;
                if (lbm.flag[i] == CellType_porous3D_Fakhari::bulk) {
                    const double P   = lbm.macro_P_only(i);
                    const double rho = lbm.local_rho(lbm.phi_cache[i]);
                    val = P * rho / 3.0;
                }
                os << static_cast<float>(val) << " ";
            }
            os << "\n";
        }
    }
    os << "\n";

    // ------------------ Velocity ------------------
    os << "VECTORS velocity float\n";
    for (int z = 0; z < dim.nz; ++z) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                const int i = idx3D(x, y, z);
                double ux = 0.0, uy = 0.0, uz = 0;
                if (lbm.flag[i] == CellType_porous3D_Fakhari::bulk) {
                    const double rho    = lbm.local_rho(lbm.phi_cache[i]);
                    const auto   u_m    = lbm.macro_u(i);

                    const auto   gphi   = lbm.gradient_phi(i);
                    const double lphi   = lbm.laplacian_phi(i);
                    const auto   grho   = lbm.gradient_rho_from_gradphi(gphi);
                    const double muphi  = lbm.mu_phi(lbm.phi_cache[i], lphi);
                    const auto   Fphi   = lbm.Fphi_fac_from(lbm.phi_cache[i], gphi);
                    const auto   Fs     = lbm.Fs_from(muphi, gphi);
                    const auto   Fb     = lbm.Fb_from(rho);
                    const double P      = lbm.macro_P_only(i);
                    const auto   Fp     = lbm.Fp_from(P, grho);

                    std::array<double,3> Ftot{ Fs[0] + Fb[0] + Fp[0], Fs[1] + Fb[1] + Fp[1], Fs[2] + Fb[2] + Fp[2] };

                    const auto u = lbm.velocity_from(u_m, Ftot, rho); // skip viscous correction if intended
                    ux = u[0]; uy = u[1]; uz = u[2];
                }
                // VECTORS velocity
                os << static_cast<float>(ux) << " " << static_cast<float>(uy) << " " << static_cast<float>(uz) << "\n";
            }
        }
    }
    os << "\n";

    // ------------------ Flag (1 = solid, 0 = fluid) ------------------
    os << "SCALARS Flag int 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int z = 0; z < dim.nz; ++z) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                const int i = idx3D(x, y, z);
                os << ((lbm.flag[i] == CellType_porous3D_Fakhari::bounce_back) ? 1 : 0) << " ";
            }
            os << "\n";
        }
    }
    os << "\n";
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ───────────────────────── Energy computation ──────────────────────────────
// Returns total kinetic energy  E = Σ ½ ρ |u|² * cell_volume
inline double computeEnergy_porous3D_Fakhari(LBM_porous3D_Fakhari& lbm,
                                             double cell_volume = 1.0)
{
    const auto& dim = lbm.dim;
    double E = 0.0;

    for (int z = 0; z < dim.nz; ++z) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                const int i = lbm.xyz_to_i(x, y, z);

                if (lbm.flag[i] == CellType_porous3D_Fakhari::bulk) {
                    const auto  u_macro = lbm.macro_u(i);
                    const double rho    = lbm.local_rho(lbm.phi_cache[i]);

                    // No forcing correction in the diagnostic:
                    const std::array<double,3> F0{0.0, 0.0, 0.0};
                    const auto u = lbm.velocity_from(u_macro, F0, rho);

                    const double u2 = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
                    E += 0.5 * rho * u2 * cell_volume;
                }
            }
        }
    }
    return E;
}
// =========================  Geometry initialization  ==========================

// ---------- Small helpers ----------
struct GeometryField {
    int nx = 0, ny = 0, nz = 0;           // lattice dims
    // 0 = fluid, 1 = solid; length = nx*ny*nz; row-major: i = z + nz*(y + ny*x)
    std::vector<uint8_t> solid;
};

inline size_t geom_xyz_to_i(int x,int y,int z,int nx,int ny,int nz){
    return static_cast<size_t>(z)
    + static_cast<size_t>(nz) * ( static_cast<size_t>(y)
    + static_cast<size_t>(ny) * static_cast<size_t>(x) );
}

inline double geometry_porosity(const GeometryField& G){
    size_t solids=0; for (auto v : G.solid) solids += (v!=0);
    return 1.0 - double(solids)/double(G.solid.size());
}

struct GeomOptions {
    double   porosity     = 0.85;   // target void fraction (0..1) on coarse interior rows
    double   mean_d       = 4.0;    // mean grain diameter in FINE LU
    double   polydisp     = 0.30;   // diameter spread (lognormal)
    unsigned seed         = 42;
    int      downscale    = 2;      // fine->coarse factor (>=1)
    bool     seal_6nb     = true;   // enforce 6-neighbor connectivity (no diagonal leaks)
    bool     periodic_x   = true;   // typical porous periodic x
    bool     periodic_z   = false;  // if false, we build hard z walls
    bool     hard_y_walls = true;   // y=0 and y=ny-1 are walls
};

// ---------- (1) 2D coarse builder (periodic x, bounded y) ----------
inline std::vector<uint8_t> build_porous_mask_coarse_2d(
    int nx, int ny,
    double target_porosity,
    double mean_grain_d_c,
    double polydisp = 0.25,
    unsigned seed   = 1337)
{
    nx = std::max(nx, 2);
    ny = std::max(ny, 3);
    target_porosity = std::clamp(target_porosity, 0.01, 0.99);
    mean_grain_d_c  = std::max(1.0, mean_grain_d_c);
    polydisp        = std::max(0.0, polydisp);

    std::vector<uint8_t> solid((size_t)nx*(size_t)ny, 0);
    auto idx   = [&](int X,int Y)->size_t { return (size_t)X + (size_t)nx*(size_t)Y; };
    auto wrapX = [&](int X)->int { int r = X % nx; return r < 0 ? r + nx : r; };

    // RNG for random discs
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const double mu_log    = std::log(mean_grain_d_c) - 0.5*std::log(1.0 + polydisp*polydisp);
    const double sigma_log = std::sqrt(std::log(1.0 + polydisp*polydisp));
    std::lognormal_distribution<double> Ddia(mu_log, sigma_log);

    auto interior_porosity = [&](){
        size_t cells=0, solid_ct=0;
        for (int Y=1; Y<=ny-2; ++Y)
            for (int X=0; X<nx; ++X){ ++cells; solid_ct += solid[idx(X,Y)] ? 1 : 0; }
            return (cells==0) ? 1.0 : std::clamp(1.0 - double(solid_ct)/double(cells), 0.0, 1.0);
    };

    const double target_phi = target_porosity;
    double phi = interior_porosity();
    int safety = nx*ny*200;

    // Recompute φ *every* disc to avoid overshoot on small grids
    while (phi > target_phi && --safety > 0) {
        int Yc = 1 + int(U01(rng) * double(ny-2));
        int Xc = int(U01(rng) * double(nx));

        double d = std::clamp(Ddia(rng), 1.0,
                              std::min<double>(std::max(nx,ny), 4.0*mean_grain_d_c));
        int R = std::max(1, int(std::round(0.5 * d)));

        int y0 = std::max(1,   Yc - R);
        int y1 = std::min(ny-2, Yc + R);

        for (int Y=y0; Y<=y1; ++Y){
            int dy = Y - Yc;
            int rx2 = R*R - dy*dy; if (rx2 < 0) continue;
            int rx  = int(std::floor(std::sqrt(double(rx2))));
            int xa  = Xc - rx;
            int xb  = Xc + rx;
            for (int X=xa; X<=xb; ++X){
                int Xw = wrapX(X);
                solid[idx(Xw,Y)] = 1;
            }
        }

        phi = interior_porosity(); // update *every* iteration
    }
    // (Optional) if (safety <= 0) std::cerr << "[coarse] hit safety cap\n";
    return solid;
}

// ---------- (2) 6-neighbor sealing (no diagonal leaks) ----------
// Only consider *interior* solids in the pair-tests. Domain walls do NOT contribute,
// otherwise everything near y/z walls gets "sealed" solid and cascades.
inline void seal_six_neighbor(
    std::vector<uint8_t>& solid,
    int nx,int ny,int nz,
    bool periodic_x, bool periodic_z,
    bool hard_y_walls, bool hard_z_walls)
{
    auto wrap = [](int a,int n){ int r=a % n; return r<0 ? r+n : r; };
    auto inside = [&](int x,int y,int z)->bool{
        return (0<=x && x<nx) && (0<=y && y<ny) && (0<=z && z<nz);
    };

    auto isDomainWall = [&](int x,int y,int z)->bool{
        if (hard_y_walls && (y==0 || y==ny-1)) return true;
        if (hard_z_walls && (z==0 || z==nz-1)) return true;
        return false;
    };

    auto isInteriorSolid = [&](int x,int y,int z)->bool{
        // apply periodic wrapping on query, but reject out-of-domain when non-periodic
        if (!periodic_x && (x<0 || x>=nx)) return false;
        if (!periodic_z && (z<0 || z>=nz)) return false;
        if (periodic_x) x = wrap(x,nx);
        if (periodic_z) z = wrap(z,nz);
        if (!inside(x,y,z)) return false;
        if (isDomainWall(x,y,z)) return false;           // <-- ignore domain walls
        return solid[ geom_xyz_to_i(x,y,z,nx,ny,nz) ] != 0;
    };

    auto markInteriorSolid = [&](int x,int y,int z){
        if (!periodic_x && (x<0 || x>=nx)) return;
        if (!periodic_z && (z<0 || z>=nz)) return;
        if (periodic_x) x = wrap(x,nx);
        if (periodic_z) z = wrap(z,nz);
        if (!inside(x,y,z)) return;
        if (isDomainWall(x,y,z)) return;                 // don't write into walls
        solid[ geom_xyz_to_i(x,y,z,nx,ny,nz) ] = 1;
    };

    auto seal_once = [&](){
        bool changed=false;
        const int zBeg = hard_z_walls ? 1 : 0;
        const int zEnd = hard_z_walls ? nz-1 : nz;
        const int yBeg = hard_y_walls ? 1 : 0;
        const int yEnd = hard_y_walls ? ny-1 : ny;

        for (int z = zBeg; z < zEnd; ++z)
            for (int y = yBeg; y < yEnd; ++y)
                for (int x = 0;   x < nx;   ++x) {
                    // skip cells that are already interior solid
                    if (isInteriorSolid(x,y,z)) continue;

                    // kill diagonal-only connectivity by sealing when *interior* orthogonal pairs block a diagonal
                    const bool xy_block =
                    (isInteriorSolid(x-1,y,z) && isInteriorSolid(x,y-1,z)) ||
                    (isInteriorSolid(x-1,y,z) && isInteriorSolid(x,y+1,z)) ||
                    (isInteriorSolid(x+1,y,z) && isInteriorSolid(x,y-1,z)) ||
                    (isInteriorSolid(x+1,y,z) && isInteriorSolid(x,y+1,z));

                    const bool xz_block =
                    (isInteriorSolid(x-1,y,z) && isInteriorSolid(x,y,z-1)) ||
                    (isInteriorSolid(x-1,y,z) && isInteriorSolid(x,y,z+1)) ||
                    (isInteriorSolid(x+1,y,z) && isInteriorSolid(x,y,z-1)) ||
                    (isInteriorSolid(x+1,y,z) && isInteriorSolid(x,y,z+1));

                    const bool yz_block =
                    (isInteriorSolid(x,y-1,z) && isInteriorSolid(x,y,z-1)) ||
                    (isInteriorSolid(x,y-1,z) && isInteriorSolid(x,y,z+1)) ||
                    (isInteriorSolid(x,y+1,z) && isInteriorSolid(x,y,z-1)) ||
                    (isInteriorSolid(x,y+1,z) && isInteriorSolid(x,y,z+1));

                    if (xy_block || xz_block || yz_block) {
                        markInteriorSolid(x,y,z);
                        changed = true;
                    }
                }
                return changed;
    };

    // One pass is enough; avoids sealing cascades.
    for (int it=0; it<1; ++it) if (!seal_once()) break;
}

// ---------- (3) Truly 3D random geometry generator ----------
inline GeometryField generate_random_geometry_3d(
    int nx,int ny,int nz,
    const GeomOptions& opt)
{
    // fine->coarse mapping
    int nx_c = std::max(2, (nx + opt.downscale - 1)/opt.downscale);
    int ny_c = std::max(3, (ny + opt.downscale - 1)/opt.downscale); // need y interior
    int nz_c = std::max(2, (nz + opt.downscale - 1)/opt.downscale);
    double mean_d_c = std::max(1.0, opt.mean_d / double(opt.downscale));

    GeometryField G{nx,ny,nz, std::vector<uint8_t>((size_t)nx*ny*nz, 0)};

    // Hard walls (optional)
    if (opt.hard_y_walls){
        for (int z=0; z<nz; ++z)
            for (int x=0; x<nx; ++x){
                G.solid[geom_xyz_to_i(x,0,      z,nx,ny,nz)]   = 1;
                G.solid[geom_xyz_to_i(x,ny-1,   z,nx,ny,nz)]   = 1;
            }
    }
    if (!opt.periodic_z){
        for (int y=0; y<ny; ++y)
            for (int x=0; x<nx; ++x){
                G.solid[geom_xyz_to_i(x,y,0,    nx,ny,nz)]     = 1;
                G.solid[geom_xyz_to_i(x,y,nz-1, nx,ny,nz)]     = 1;
            }
    }

    // Build independent coarse masks per coarse z
    std::vector<std::vector<uint8_t>> coarse(nz_c);
    for (int zc=0; zc<nz_c; ++zc){
        unsigned seed_z = opt.seed + 1013u*unsigned(zc);
        coarse[zc] = build_porous_mask_coarse_2d(
            nx_c, ny_c, opt.porosity, mean_d_c, opt.polydisp, seed_z
        );
    }

    // Upsample interior
    const double sx = double(nx_c)        / double(std::max(1,nx));
    const double sy = double(ny_c - 2)    / double(std::max(1, ny - 2)); // exclude y-walls
    const double sz = double(nz_c)        / double(std::max(1, nz - 2)); // exclude z-walls

    auto idx_c = [&](int Xc,int Yc){ return (size_t)Xc + (size_t)nx_c*(size_t)Yc; };

    int z0 = (!opt.periodic_z ? 1 : 0);
    int z1 = (!opt.periodic_z ? nz-1 : nz);
    int y0 = (opt.hard_y_walls ? 1 : 0);
    int y1 = (opt.hard_y_walls ? ny-1 : ny);

    for (int z = z0; z < z1; ++z){
        int zc = std::min(nz_c-1, int(std::floor((z - z0) * sz)));
        const auto& S = coarse[zc];
        for (int y = y0; y < y1; ++y){
            int yc = 1 + int(std::floor((y - y0) * sy));
            yc = std::clamp(yc, 1, ny_c-2);
            for (int x = 0; x < nx; ++x){
                int xc = std::min(nx_c-1, int(std::floor(x * sx)));
                if (S[idx_c(xc,yc)]){
                    G.solid[ geom_xyz_to_i(x,y,z,nx,ny,nz) ] = 1;
                }
            }
        }
    }

    // (debug) report before sealing
    // std::cout << "[geom] coarse-upsample porosity(before seal) = " << geometry_porosity(G) << "\n";

    // Sealing (remove diagonal leaks) — walls ignored inside pair tests
    if (opt.seal_6nb){
        seal_six_neighbor(G.solid, nx, ny, nz,
                          /*periodic_x=*/opt.periodic_x,
                          /*periodic_z=*/opt.periodic_z,
                          /*hard_y_walls=*/opt.hard_y_walls,
                          /*hard_z_walls=*/!opt.periodic_z);
    }

    // (debug) report after sealing
    // std::cout << "[geom] porosity(after seal) = " << geometry_porosity(G) << "\n";

    return G;
}

// ---------- (4) ASCII geometry reader ----------
inline GeometryField read_geometry_ascii(const std::string& path)
{
    std::ifstream is(path);
    if (!is) throw std::runtime_error("Cannot open geometry file: " + path);

    int nx=0, ny=0, nz=0;
    {
        // skip comments starting with '#'
        std::string header;
        do {
            if (!std::getline(is, header))
                throw std::runtime_error("Geometry file: missing header nx ny nz");
        } while (!header.empty() && header[0] == '#');

        std::istringstream hs(header);
        if (!(hs >> nx >> ny >> nz) || nx<=0 || ny<=0 || nz<=0)
            throw std::runtime_error("Bad header (nx ny nz) in geometry file.");
    }

    GeometryField G{nx,ny,nz, std::vector<uint8_t>((size_t)nx*ny*nz, 0)};

    auto put_line = [&](const std::string& line, int y, int z){
        // Accept "010101", "0 1 0 1 0 1", or mixed whitespace
        int x = 0;
        for (char c : line){
            if (c=='0' || c=='1'){
                if (x >= nx) break;
                G.solid[ geom_xyz_to_i(x,y,z,nx,ny,nz) ] = (c=='1') ? 1 : 0;
                ++x;
            }
        }
        // Pad remaining as fluid if short
        for (; x<nx; ++x)
            G.solid[ geom_xyz_to_i(x,y,z,nx,ny,nz) ] = 0;
    };

    std::string line;
    for (int z=0; z<nz; ++z){
        for (int y=0; y<ny; ++y){
            // Skip comment/empty lines
            do {
                if (!std::getline(is, line))
                    throw std::runtime_error(
                        "Unexpected EOF while reading geometry (z="
                        + std::to_string(z) + ", y=" + std::to_string(y) + ")");
            } while (line.empty() || line[0] == '#');
            put_line(line, y, z);
        }
    }
    return G;
}

// ---------- (5) Apply a GeometryField into your LBM object ----------
inline void apply_geometry_to_lbm(LBM_porous3D_Fakhari& lbm, const GeometryField& G)
{
    if (lbm.dim.nx != G.nx || lbm.dim.ny != G.ny || lbm.dim.nz != G.nz)
        throw std::runtime_error(
            "apply_geometry_to_lbm: dimension mismatch. Recreate LBM with geometry dims.");

        for (int x=0; x<lbm.dim.nx; ++x)
            for (int y=0; y<lbm.dim.ny; ++y)
                for (int z=0; z<lbm.dim.nz; ++z){
                    const size_t gi = geom_xyz_to_i(x,y,z,G.nx,G.ny,G.nz);
                    const int    li = (int)lbm.xyz_to_i(x,y,z);
                    if (G.solid[gi]) {
                        lbm.flag[li] = CellType_porous3D_Fakhari::bounce_back;
                        for (int k=0;k<19;++k){ lbm.fin(li,k)=0.0; lbm.gin(li,k)=0.0; }
                    } else {
                        lbm.flag[li] = CellType_porous3D_Fakhari::bulk;
                    }
                }
}




// ─────────────────────────   main driver (Rayleigh–Taylor, TILED TBB)  ─────
inline void porous3D_Fakhari()
{
    // ------------------- Config -------------------
    std::ifstream contfile("../apps/Config_Files/config_porous3D_Fakhari.txt");
    if (!contfile.is_open()) {
        throw std::invalid_argument(
            "Config file not found: ../apps/Config_Files/config_porous3D_Fakhari.txt");
    }

    // Defaults (override via config)
    double ulb=1.0, max_t=0.01, phi_l=1.0, phi_g=0.0, rho_l=1.0, rho_g=1e-3;
    double Sigma=0.01, xi=4.0, M=0.02, Fbx=0.0, Fby=0.0, Fbz = 0.;
    double mu_l=0.1, mu_g=1e-4;
    int N=128, out_freq=100, vtk_freq=100;

    std::string line, param, value;
    while (std::getline(contfile, line)) {
        if (auto pos = line.find('#'); pos != std::string::npos) line.erase(pos);
        if (line.empty()) continue;
        std::istringstream ls(line);
        if (!(ls >> param >> value)) continue;

        if      (param == "N")        N       = std::stoi(value);
        else if (param == "max_t")    max_t   = std::stod(value);
        else if (param == "out_freq") out_freq= std::stoi(value);
        else if (param == "vtk_freq") vtk_freq= std::stoi(value);
        else if (param == "phi_l")    phi_l   = std::stod(value);
        else if (param == "phi_g")    phi_g   = std::stod(value);
        else if (param == "rho_l")    rho_l   = std::stod(value);
        else if (param == "rho_g")    rho_g   = std::stod(value);
        else if (param == "Sigma")    Sigma   = std::stod(value);
        else if (param == "xi")       xi      = std::stod(value);
        else if (param == "M")        M       = std::stod(value);
        else if (param == "ulb")      ulb     = std::stod(value);
        else if (param == "Fbx")      Fbx     = std::stod(value);
        else if (param == "Fby")      Fby     = std::stod(value);
        else if (param == "Fbz")      Fbz     = std::stod(value);
        else if (param == "mu_l")     mu_l    = std::stod(value);
        else if (param == "mu_g")     mu_g    = std::stod(value);
        else {
            std::cerr << "Warning: unknown parameter \"" << param << "\"\n";
        }
    }

    GeomOptions opt;
    opt.porosity   = 0.8;
    opt.mean_d     = 5.0;
    opt.polydisp   = 0.30;
    opt.seed       = 42;
    opt.downscale  = 3;
    opt.seal_6nb   = true;
    opt.periodic_x = true;
    opt.periodic_z = true;
    opt.hard_y_walls = true;



// --------------------------- OR if you have ageometry file----------------------------------------------
    // 1) Read geometry from file:
    // GeometryField G = read_geometry_ascii("../apps/geometry/my_medium.geom");

    // 2) Construct dim with the geometry sizes:
    // Dim_porous3D_Fakhari dim { G.nx, G.ny, G.nz };

    // ------------------- Lattice dims & params -------------------
    // Example 3D domain: elongated in x, hard walls in y, periodic x/z (as used in your geom builder)
    Dim_porous3D_Fakhari dim { 2*N, N+2, N+2 };

    auto [Re, At, Ca, Pe, dx, dt] =
    lbParameters_porous3D_Fakhari(ulb, N, mu_l, Sigma, rho_l, rho_g, M);
    printParameters_porous3D_Fakhari(dim, Re, At, Ca, Pe, ulb, N, max_t);

    // ------------------- Storage -------------------
    std::vector<CellData> lattice_vect(LBM_porous3D_Fakhari::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();

    std::vector<CellType_porous3D_Fakhari> flag_vect(dim.nelem);
    auto* flag = flag_vect.data();

    std::vector<int> parity_vect(1, 0);
    int* parity = parity_vect.data();

    std::vector<int> nb_vect(dim.nelem * 19), nbb_vect(dim.nelem * 19);
    std::vector<double> phi_vect(dim.nelem);

    auto [c_vect, opp_vect, t_vect] = d3q19_constants_porous3D_Fakhari();

    LBM_porous3D_Fakhari lbm{
        lattice, flag, parity,
        c_vect.data(), opp_vect.data(), t_vect.data(),
        /* nb  */ nullptr,
        /* nbb */ nullptr,
        /* phi_cache */ nullptr,
        phi_l, phi_g, rho_l, rho_g,
        mu_l, mu_g, Sigma,
        Fbx, Fby, Fbz, xi, M,
        dim
    };

    // Bind neighbor tables & cache
    lbm.nb        = nb_vect.data();
    lbm.nbb       = nbb_vect.data();
    lbm.phi_cache = phi_vect.data();
    lbm.build_neighbors();

    //inigeom_porous3D_Fakhari(lbm, porosity, grain_d, polydisp, seed, downscale);
    GeometryField G = generate_random_geometry_3d(dim.nx, dim.ny, dim.nz, opt);
    std::cout << "[geom] porosity(before apply) = " << geometry_porosity(G) << "\n";
    apply_geometry_to_lbm(lbm, G);


    size_t solids=0;
    for (auto s: G.solid) solids += s!=0;
    double por = 1.0 - double(solids)/double(G.solid.size());
    std::cout << "[geom] solids=" << solids
    << "  porosity=" << por
    << "  dims=" << G.nx << "x" << G.ny << "x" << G.nz << "\n";


    // ------------------- Initialization -------------------
    std::for_each(lattice, lattice + dim.nelem,
                  [&lbm](CellData& f0) { lbm.iniLattice(f0); });

    auto [start, clock_iter] = restartClock_porous3D_Fakhari();
    std::ofstream efile("energy.dat");

    // ------------------- TBB setup -------------------
    int hw = oneapi::tbb::info::default_concurrency();
    int threads = hw;
    if (const char* env = std::getenv("COOLBM_THREADS")) {
        if (int req = std::atoi(env); req > 0 && req <= hw) threads = req;
    }
    oneapi::tbb::global_control gc(oneapi::tbb::global_control::max_allowed_parallelism, threads);
    oneapi::tbb::task_arena arena(threads);
    oneapi::tbb::affinity_partitioner ap;

    // Tile sizes (3D)
    const int TX = std::max(8, std::min(32, dim.nx/2));
    const int TY = std::max(8, std::min(32, dim.ny/2));
    const int TZ = std::max(4, std::min(16, dim.nz/2));


    const int max_time_iter = static_cast<int>(max_t / dt);

    arena.execute([&]{
        for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {

            // 1) build φ cache (3D)
            lbm.build_phi_cache_tiled(TX, TY, TZ);

            // 2) collide + stream (3D tiles)
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range3d<int>(
                    0, dim.nz, TZ,   // pages (z)
                0, dim.ny, TY,   // rows  (y)
                0, dim.nx, TX    // cols  (x)
                ),
                [&](const oneapi::tbb::blocked_range3d<int>& r){
                    for (int z = r.pages().begin(); z < r.pages().end(); ++z) {
                        for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
                            for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                                const int i = static_cast<int>(lbm.xyz_to_i(x,y,z));
                                lbm(lbm.lattice[i]);
                            }
                        }
                    }
                },
                ap
            );

            // Output (do after collide+stream if you want “post” state; or before if “pre”)
            if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
                saveVtkFields_porous3D_Fakhari(lbm, time_iter);
            }
            if (out_freq != 0 && time_iter % out_freq == 0) {
                std::cout << "Saving profiles at iteration " << time_iter
                << ", t = " << std::setprecision(6) << time_iter * dt << std::setprecision(3)
                << " [" << (time_iter * dt / max_t * 100.) << "%]\n";

                // φ cache already built above; rebuild only if you output pre-collision state.
                const double cell_vol = (1.0/dim.nx) * (1.0/dim.nx) * (1.0/dim.nx); // 3D volume in LU
                const double energy   = computeEnergy_porous3D_Fakhari(lbm, cell_vol);
                std::cout << "Total kinetic energy: " << std::setprecision(8) << energy << "\n";
                efile << std::setw(12) << std::setprecision(6) << time_iter * dt
                << std::setw(18) << std::setprecision(8) << energy << "\n";
            }

            // 3) swap buffers
            *parity = 1 - *parity;
            ++clock_iter;
        }
    });

    printMlups_porous3D_Fakhari(start, clock_iter, dim.nelem);
}

