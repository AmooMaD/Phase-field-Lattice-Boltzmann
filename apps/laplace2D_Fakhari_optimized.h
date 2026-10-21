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

// ─────────────────────────   cell-level helpers   ───────────────────────────
enum class CellType_rayleighTaylor2D_Fakhari : uint8_t { bounce_back, bulk };
using  CellData = double;

// ─────────────────────────   D2Q9 constants   ───────────────────────────────
inline auto d2q9_constants_rayleighTaylor2D_Fakhari()
{
    // k=4 is the rest (center)
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
struct Dim_rayleighTaylor2D_Fakhari {
    Dim_rayleighTaylor2D_Fakhari(int nx_, int ny_)
    : nx(nx_), ny(ny_),
    nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)),
    npop (9 * nelem) {}
    int    nx, ny;
    size_t nelem, npop;
};

// ─────────────────────────   lattice-unit parameters   ─────────────────────
inline auto lbParameters_rayleighTaylor2D_Fakhari(double ulb, int lref, double mu_l, double Sigma, double rho_l, double rho_g, double M)
{
    double Re   = rho_l * ulb * static_cast<double>(lref) / mu_l;
    double At   = (rho_l - rho_g ) / (rho_l + rho_g);
    double Ca   = mu_l * ulb / Sigma;
    double Pe   = ulb * static_cast<double>(lref) / M;
    double dx   = 1. / static_cast<double>(lref);
    double dt   = dx * ulb;
    return make_tuple(Re, At, Ca, Pe, dx, dt);
}

inline void printParameters_rayleighTaylor2D_Fakhari(const Dim_rayleighTaylor2D_Fakhari &dim,
                                                     double Re, double At, double Ca, double Pe, double ulb,
                                                     int N, double max_t)
{
    cout << "Rayleigh–Taylor 2-D (Conservative PF — BGK/BGK)\n"
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
inline auto restartClock_rayleighTaylor2D_Fakhari()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_rayleighTaylor2D_Fakhari(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    int dure = static_cast<int>(us.count());
    cout << "Runtime: " << dure/1000000. << " seconds " << endl;
    cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

// ─────────────────────────   LBM functor   ─────────────────────────────────
struct LBM_rayleighTaylor2D_Fakhari
{
    using  CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem + 2 * 9 * nelem; }

    // raw buffers
    CellData*                        lattice;
    CellType_rayleighTaylor2D_Fakhari* flag;
    int*                             parity;
    std::array<int,2>*               c;
    int*                             opp;
    double*                          t;

    // neighbor tables (precomputed)
    int* nb    = nullptr; // size = nelem*9
    int* nbb   = nullptr; // mirror index for bounce-back along k (size = nelem*9)

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
    Dim_rayleighTaylor2D_Fakhari dim;

    // constants
    static constexpr double cs2 = 1.0/3.0;
    static constexpr double PI  = 3.1415926535897932384626433832795;

    // ───────── index helpers ─────────
    inline size_t xyz_to_i (int x, int y) const {
        return (size_t)(y + dim.ny * x);
    }
    inline auto i_to_xyz (int i) const {
        int iX = i / (dim.ny);
        int iY = i % (dim.ny);
        return std::make_tuple(iX, iY);
    }

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
        // expects nb and nbb pointers to be valid (pointing to storage of size nelem*9)
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

    // ───────────────────────── initialization (BENCH: droplet, gin=0) ──────────────────────────────
    /*inline void iniLattice(double& f0)
    {
        auto i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz(static_cast<int>(i));

        // --- droplet geometry (circle) ---
        double xc = 0.5 * double(dim.nx);
        double yc = 0.5 * double(dim.ny);
        double R0 = double(dim.ny) / 4.0;   // radius (LU)

        // cell-center coordinates
        double X = double(iX) + 0.5;
        double Y = double(iY) + 0.5;

        // periodic minimal-image shift to center
        double dx = X - xc;
        double dy = Y - yc;
        (void)dx; (void)dy; // used via hypot below; keep to silence -Wunused if optimized
        double phi0 = 0.5*(phi_l + phi_g);
        double amp  = 0.5*(phi_l - phi_g);
        double phi  = phi0 + amp * std::tanh( (R0 - std::hypot(dx,dy)) / (0.5*xi) );

        // populate both distribution sets with equilibria at U=0
        for (int k = 0; k < 9; ++k) {
            fin((int)i,k) = phi * t[k];
            gin((int)i,k) = 0.0; // P* = 0 at t=0 (matches Laplace-style init)
        }
    }
    */

    inline void iniLattice(double& f0) {
        auto i = static_cast<int>(&f0 - lattice);
        auto [iX, iY] = i_to_xyz(i);

        double xc = 0.5 * double(dim.nx);
        double yc = 0.5 * double(dim.ny);
        double R0 = double(dim.ny) / 4.0;   // half side length of the square

        double X = double(iX) + 0.5;
        double Y = double(iY) + 0.5;

        double dx = X - xc;
        double dy = Y - yc;

        double phi0 = 0.5 * (phi_l + phi_g);
        double amp  = 0.5 * (phi_l - phi_g);

        // square signed distance approximation: max(|dx|,|dy|) - R0
        double d = std::max(std::abs(dx), std::abs(dy));
        double phi = phi0 + amp * std::tanh((R0 - d) / (0.5 * xi));

        for (int k = 0; k < 9; ++k) {
            fin(i,k) = phi * t[k];
            gin(i,k) = 0.0;
        }
    }


    // ───────────────────────── kernels ─────────────────────────

    // macro-vars ϕ, P from current (in) populations
    inline auto macro_phi_P(int i) const {
        // f-sum (phi)
        double phi =
        (fin(i,0)+fin(i,2)+fin(i,3)) +
        (fin(i,5)+fin(i,7)+fin(i,8)) +
        (fin(i,6)+fin(i,1)+fin(i,4));
        // g-sum (P*)
        double P =
        (gin(i,0)+gin(i,2)+gin(i,3)) +
        (gin(i,5)+gin(i,7)+gin(i,8)) +
        (gin(i,6)+gin(i,1)+gin(i,4));
        return std::make_tuple(phi, P);
    }

    // phi-only for stencils (avoid touching g when unnecessary)
    inline double macro_phi_only(int i) const {
        return (fin(i,0)+fin(i,2)+fin(i,3)) +
        (fin(i,5)+fin(i,7)+fin(i,8)) +
        (fin(i,6)+fin(i,1)+fin(i,4));
    }

    // Eq. 32b first term (without force shift)
    inline auto macro_u(int i) const {
        array<double,2> u{};
        double Xg_M1 = gin(i, 0) + gin(i, 2) + gin(i, 3);
        double Xg_P1 = gin(i, 5) + gin(i, 7) + gin(i, 8);
        double Yg_P1 = gin(i, 3) + gin(i, 7) + gin(i, 6);
        double Yg_M1 = gin(i, 2) + gin(i, 1) + gin(i, 8);
        u[0] = Xg_P1 - Xg_M1;
        u[1] = Yg_P1 - Yg_M1;
        return u;
    }

    inline double local_rho(double phi) const {
        return rho_g + (phi - phi_g) * (rho_l - rho_g); //rho_g + ((phi - phi_g)/(phi_l - phi_g)) * (rho_l - rho_g);
    }
    inline double local_mu(double phi) const {
        return mu_g + (phi - phi_g) * (mu_l - mu_g); //mu_g + ((phi - phi_g)/(phi_l - phi_g)) * (mu_l - mu_g);
    }
    inline double local_tau(double rho, double mu) const { return mu / (rho * cs2); }
    inline double local_nu (double tau) const { return tau * cs2; }

    // ∇phi (Eq. 34) using neighbor tables + φ-only
    inline std::array<double,2> gradient_phi(int i) const {
        double gx=0.0, gy=0.0;
        const int base = i*9;
        for (int k=0;k<9;++k) {
            int j = nb[base + k];
            if (flag[j] == CellType_rayleighTaylor2D_Fakhari::bounce_back) {
                j = nbb[base + k];
            }
            double phi_nb = macro_phi_only(j);
            gx += t[k] * c[k][0] * phi_nb;
            gy += t[k] * c[k][1] * phi_nb;
        }
        return { gx / cs2, gy / cs2};
    }

    // ∇²phi (Eq. 35) using neighbor tables + φ-only
    inline double laplacian_phi(int i) const {
        const double phi_c = macro_phi_only(i);
        double sum = 0.0;
        const int base = i*9;
        for (int k=0;k<9;++k) {
            int j = nb[base + k];
            if (flag[j] != CellType_rayleighTaylor2D_Fakhari::bounce_back) {
                double phi_n = macro_phi_only(j);
                sum += t[k] * (phi_n - phi_c);
            }
        }
        return 2. * sum / cs2;
    }

    // ∇ρ via ∇φ
    inline std::array<double,2> gradient_rho_from_gradphi(const std::array<double,2>& grad_phi) const {
        double delta_rho = rho_l - rho_g;
        return { delta_rho * grad_phi[0], delta_rho * grad_phi[1] };
    }

    // μ_φ (Eq. 5)
    inline double mu_phi(double phi, double lap_phi) const {
        double beta  = 12.0 * Sigma / xi;
        double kappa = 3. * Sigma * xi / 2.;
        double phi0  = (phi_l + phi_g)/2.;
        return 4 * beta * (phi - phi_g) * (phi - phi_l) * (phi - phi0) - kappa * lap_phi;
    }

    // Forces (Fs, Fb, Fp, Fmu) — take precomputed inputs to avoid recomputation
    inline std::array<double,2> Fs_from(double mu_phi_val, const std::array<double,2>& grad_phi) const {
        return { mu_phi_val * grad_phi[0], mu_phi_val * grad_phi[1] };
    }
    inline std::array<double,2> Fb_from(double rho) const {
        // uniform body force (change to buoyancy form if desired)
        return { rho * Fbx, rho * Fby };
    }
    inline std::array<double,2> Fp_from(double Pstar, const std::array<double,2>& grad_rho) const {
        return { -Pstar * cs2 * grad_rho[0], -Pstar * cs2 * grad_rho[1] };
    }


    inline std::array<double,2> Total_Fsbp(const std::array<double,2>& Fs_from, 
                                           const std::array<double,2>& Fb_from,
                                           const std::array<double,2>& Fp_from){

        return {Fs_from[0] + Fb_from[0] + Fp_from[0] , Fs_from[1] + Fb_from[1] + Fp_from[1]};
    }

    std::array<double,2> mid_velocity(const std::array<double,2>& u_macro,
                                      const std::array<double,2>& Total_Fsbp, 
                                      double rho){  // Eq. 32b

        return{u_macro[0] + Total_Fsbp[0]/(2.*rho), u_macro[1] + Total_Fsbp[1]/(2.*rho)};
    }

    inline std::array<double,2> Fmu_from(int i,
                                         const std::array<double,2>& u,
                                         double pstar, double tau, double nu,
                                         const std::array<double,2>& grad_rho) const
                                         {
                                             double usqr = (u[0]*u[0] + u[1]*u[1]) / (2.0*cs2);
                                             double fac  = - nu / ((tau + 0.5) * cs2);

                                             double Sxx = 0.0, Sxy = 0.0, Syy = 0.0;
                                             for (int k=0; k<9; ++k) {
                                                if (k==4) continue;

                                                 double ex = (double)c[k][0];
                                                 double ey = (double)c[k][1];
                                                 double ck_u = ex*u[0] + ey*u[1];
                                                 double gamma_k = t[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - usqr);
                                                 double geq_k   = pstar * t[k] + (gamma_k - t[k]);
                                                 double gneq    = gin(i,k) - geq_k;
                                                 Sxx += gneq * ex * ex;
                                                 Sxy += gneq * ex * ey;
                                                 Syy += gneq * ey * ey;
                                             }

                                             double Fx = fac * ( Sxx * grad_rho[0] + Sxy * grad_rho[1] );
                                             double Fy = fac * ( Sxy * grad_rho[0] + Syy * grad_rho[1] );
                                             return {Fx, Fy};
                                         }

    // Velocity with force shift (Eq. 32b)
    inline std::array<double,2> velocity_from(const std::array<double,2>& u_macro,
                                           const std::array<double,2>& F_total,
                                           double rho) const
                                           {
                                               return { u_macro[0] + F_total[0]/(2.*rho),
                                                   u_macro[1] + F_total[1]/(2.*rho) };
                                           }

    inline std::array<double,2> Fphi_fac_from(double phi, const std::array<double,2>& grad_phi) const {
       double phi0 = (phi_g + phi_l) / 2.;
       double fac  = (1 - 4*(phi - phi0)*(phi - phi0)) / xi;
       double s    = std::sqrt(grad_phi[0]*grad_phi[0] + grad_phi[1]*grad_phi[1]) + 1e-30;
       return { fac * grad_phi[0] / s, fac * grad_phi[1] / s };
    }

    // Stream to neighbor or bounce-back mirror (using tables)
    inline void stream_tab(int i,int k,double pf,double pg)
    {
       const int j = nb[i*9 + k];
       if (flag[j] == CellType_rayleighTaylor2D_Fakhari::bounce_back) {
           fout(i, opp[k]) = pf;
           gout(i, opp[k]) = pg;
       } else {
           fout(j, k) = pf;
           gout(j, k) = pg;
       }
    }

    // BGK collision for the pair (f,g)
    inline std::tuple<double,double,double,double>
    collideBgk(int i,int k,
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

       double feq_bar      = phi * gamma      - F_f / 2.;   // Eq. 9
       double feq_bar_opp  = phi * gamma_opp  + F_f / 2.;

       double pf     = fin(i,k)      - (fin(i,k)      - feq_bar     ) / (0.5 + tau_phi) + F_f;
       double pf_opp = fin(i,opp[k]) - (fin(i,opp[k]) - feq_bar_opp ) / (0.5 + tau_phi) - F_f;

       double F_g    = /*(1 - 0.5/tau)* */ t[k] * (F[0]*c[k][0] + F[1]*c[k][1]) / (rho * cs2);

       double geq_bar     = geq     - F_g/2.;
       double geq_bar_opp = geq_opp + F_g/2.;

       double pg     = gin(i,k)      - (gin(i,k)      - geq_bar    ) / (0.5 + tau) + F_g;
       double pg_opp = gin(i,opp[k]) - (gin(i,opp[k]) - geq_bar_opp) / (0.5 + tau) - F_g;

       return {pf, pg, pf_opp, pg_opp};
    }

    // main per-cell operator — optimized to avoid recomputations
    inline void operator()(double& f0)
    {
       int i = (int)(&f0 - lattice);
       if (flag[i] != CellType_rayleighTaylor2D_Fakhari::bulk) return;

       // Precompute all macros once
       auto [phi, Pstar] = macro_phi_P(i);
       double rho  = local_rho(phi);
       double mu   = local_mu(phi);
       double tau  = local_tau(rho, mu);
       double nu   = local_nu(tau);
       auto u_macro = macro_u(i);

       auto grad_phi = gradient_phi(i);
       double lap_phi = laplacian_phi(i);
       auto grad_rho  = gradient_rho_from_gradphi(grad_phi);

       // μ_φ and F_phi
       double mu_phi_val = mu_phi(phi, lap_phi);
       auto F_phi = Fphi_fac_from(phi, grad_phi);

       // Forces — each uses precomputed inputs
       auto Fs  = Fs_from(mu_phi_val, grad_phi);
       auto Fb  = Fb_from(rho);
       auto Fp  = Fp_from(Pstar, grad_rho);

       auto F_sbp = Total_Fsbp(Fs,Fb,Fp);
       auto u_mid = mid_velocity(u_macro, F_sbp, rho) ;
       auto Fmu = Fmu_from(i, u_mid, Pstar, tau, nu, grad_rho);

       std::array<double,2> F_total { Fs[0] + Fb[0] + Fp[0] + Fmu[0],
           Fs[1] + Fb[1] + Fp[1] + Fmu[1] };

           // Physical velocity (with force shift)
           auto u = velocity_from(u_macro, F_total, rho);
           double usqr = (u[0]*u[0] + u[1]*u[1]) / ( 2. * cs2 );

           // Stream the 8 moving directions in pairs (k and opp[k]) via tables
           for (int k = 0; k < 4; ++k) {
               auto [pf, pg, pf_op, pg_op] = collideBgk(
                   i, k, u, rho, phi, Pstar, usqr, tau, F_total, F_phi
               );
               stream_tab(i, k     , pf   , pg   );
               stream_tab(i, opp[k], pf_op, pg_op);
           }

           // Rest population (k = 4)
           {
               int k = 4;
               double tau_phi = M / cs2;

               double feq0 = phi * t[k] * (1. - usqr);
               double geq0 = t[k] * (Pstar - usqr);

               double geq_bar0 = geq0; // Fg = 0

               fout(i,k) = fin(i,k) - (fin(i,k) - feq0) / (tau_phi + 0.5);
               gout(i,k) = gin(i,k) - (gin(i,k) - geq_bar0) / (tau + 0.5);
           }
    }
};

// ───────────────────────── VTK output ─────────────────────────────────
inline void saveVtkFields_rayleighTaylor2D_Fakhari(LBM_rayleighTaylor2D_Fakhari& lbm, int time_iter, double dx = 0.) {
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

    // φ
    os << "SCALARS phi float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            float val = 0.f;
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                val = (float)lbm.macro_phi_only(i);
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
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                double phi = lbm.macro_phi_only(i);
                val = (float)lbm.local_rho(phi);
            }
            os << val << " ";
        }
        os << "\n";
    }
    os << "\n";

    // p* * rho / 3 (diagnostic)
    os << "SCALARS pressure float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double val = 0.;
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                auto [phi, p] = lbm.macro_phi_P(i);
                double rho = lbm.local_rho(phi);
                val = p * rho / 3.;
            }
            os << val << " ";
        }
        os << "\n";
    }
    os << "\n";

    // Velocity (vector field) — cheap diagnostic (skip Fμ)
    os << "VECTORS velocity float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double ux = 0., uy = 0.;
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                auto [phi, P] = lbm.macro_phi_P(i);
                double rho    = lbm.local_rho(phi);
                auto u_macro  = lbm.macro_u(i);

                auto grad_phi = lbm.gradient_phi(i);
                double lap_phi= lbm.laplacian_phi(i);
                auto grad_rho = lbm.gradient_rho_from_gradphi(grad_phi);
                double mu_phi_val = lbm.mu_phi(phi, lap_phi);
                auto F_phi  = lbm.Fphi_fac_from(phi, grad_phi);
                auto Fs     = lbm.Fs_from(mu_phi_val, grad_phi);
                auto Fb     = lbm.Fb_from(rho);
                auto Fp     = lbm.Fp_from(P, grad_rho);
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
            os << ((lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bounce_back) ? 1 : 0) << " ";
        }
        os << "\n";
    }
    os << "\n";
}

// ───────────────────────── Energy computation ──────────────────────────────
inline double computeEnergy_rayleighTaylor2D_Fakhari(LBM_rayleighTaylor2D_Fakhari& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_rayleighTaylor2D_Fakhari::bulk) {
                auto [phi, P] = lbm.macro_phi_P(i);
                double rho    = lbm.local_rho(phi);
                auto u_macro  = lbm.macro_u(i);
                std::array<double,2> F0{0,0};
                auto u = lbm.velocity_from(u_macro, F0, rho);
                energy += u[0]*u[0] + u[1]*u[1];
            }
        }
    }
    return 0.5 * energy;
}

// ───────────────────────── Geometry initialization ─────────────────────────
/*inline void inigeom_rayleighTaylor2D_Fakhari(LBM_rayleighTaylor2D_Fakhari& lbm) {
    Dim_rayleighTaylor2D_Fakhari const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            // Example: periodic domain (all bulk). For walls, set y==0/ny-1 to bounce_back.
            lbm.flag[i] = CellType_rayleighTaylor2D_Fakhari::bulk;
        }
    }
} */

inline void inigeom_rayleighTaylor2D_Fakhari(LBM_rayleighTaylor2D_Fakhari& lbm)
{
    Dim_rayleighTaylor2D_Fakhari const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (y == -11|| y == dim.ny + 1) {
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


// ─────────────────────────   main driver (Rayleigh–Taylor)  ────────────────────────
inline void rayleighTaylor2D_Fakhari()
{
    ifstream contfile("../apps/Config_Files/config_laplace2D_Fakhari.txt");
    if (!contfile.is_open()) {
        contfile.open("../apps/Config_Files/config_laplace2D_Fakhari.txt");
        if (!contfile.is_open()) {
            throw invalid_argument(
                "Config file not found. Expected "
                "\"config_rayleighTaylor2D_Fakhari.txt\" (or legacy \"config_laplace2D_Fakhari.txt\") in apps/Config_Files.");
        }
    }

    // Parameters (defaults)
    double ulb=1.0, max_t=0.01, phi_l=1.0, phi_g=0.0, rho_l=1.0, rho_g=1e-3;
    double Sigma=0.01, xi=4.0, M=0.02, Fbx=0.0, Fby=0.0;
    double mu_l=0.1, mu_g=0.0001;
    int N=128, out_freq=100, vtk_freq=100;

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
        else if (param == "xi")         xi      = stod(value);
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

    // lattice parameters
    Dim_rayleighTaylor2D_Fakhari dim {N, N}; // square domain for the droplet benchmark
    auto [Re, At, Ca, Pe, dx, dt] =
    lbParameters_rayleighTaylor2D_Fakhari(ulb, N, mu_l, Sigma, rho_l , rho_g , M);
    printParameters_rayleighTaylor2D_Fakhari(dim, Re, At, Ca, Pe, ulb, N, max_t);

    // allocate distributions
    vector<CellData> lattice_vect(LBM_rayleighTaylor2D_Fakhari::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();

    // flags
    vector<CellType_rayleighTaylor2D_Fakhari> flag_vect(dim.nelem);
    CellType_rayleighTaylor2D_Fakhari* flag = &flag_vect[0];

    // parity
    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];

    // neighbors
    vector<int> nb_vect(dim.nelem*9), nbb_vect(dim.nelem*9);

    // constants
    auto [c_vect, opp_vect, t_vect] = d2q9_constants_rayleighTaylor2D_Fakhari();

    LBM_rayleighTaylor2D_Fakhari lbm{
        lattice,                           // lattice
        flag,                              // flag
        parity,                            // parity
        c_vect.data(),                     // c
        opp_vect.data(),                   // opp
        t_vect.data(),                     // t
        /* nb  */ nullptr,                 // <--- add this
        /* nbb */ nullptr,                 // <--- and this
        phi_l, phi_g, rho_l, rho_g,        // model params...
        mu_l, mu_g, Sigma,
        Fbx, Fby, xi, M,
        dim
    };

    // then bind the neighbor tables you allocated above:
    lbm.nb  = nb_vect.data();
    lbm.nbb = nbb_vect.data();
    lbm.build_neighbors();


    // init fields
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });
    inigeom_rayleighTaylor2D_Fakhari(lbm);

    auto [start, clock_iter] = restartClock_rayleighTaylor2D_Fakhari();
    ofstream efile("energy.dat");

    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
            saveVtkFields_rayleighTaylor2D_Fakhari(lbm, time_iter);
        }
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter
            << ", t = " << setprecision(6) << time_iter * dt << setprecision(3)
            << " [" << (time_iter * dt / max_t * 100.) << "%]" << endl;
            double energy = computeEnergy_rayleighTaylor2D_Fakhari(lbm) * dx * dx / (dt * dt);
            cout << "Average energy: " << setprecision(8) << energy << endl;
            efile << setw(12) << setprecision(6) << time_iter * dt
            << setw(18) << setprecision(8) << energy << "\n";
        }

        // main step — parallelized over cells
        //for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        // swap buffers
        *parity = 1 - *parity;
        ++clock_iter;
    }

    printMlups_rayleighTaylor2D_Fakhari(start, clock_iter, dim.nelem);
}
