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
enum class CellType_layered2D_Fakhari : uint8_t { bounce_back, bulk };
using  CellData = double;

// ─────────────────────────   D2Q9 constants   ───────────────────────────────
inline auto d2q9_constants_layered2D_Fakhari()
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
struct Dim_layered2D_Fakhari {
    Dim_layered2D_Fakhari(int nx_, int ny_)
    : nx(nx_), ny(ny_),
    nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)),
    npop (9 * nelem) {}
    int    nx, ny;
    size_t nelem, npop;
};

// ─────────────────────────   lattice-unit parameters   ─────────────────────
inline auto lbParameters_layered2D_Fakhari(double ulb, int lref, double mu_l, double Sigma, double rho_l, double rho_g, double M)
{
    double Re   = rho_l * ulb * static_cast<double>(lref) / mu_l;
    double At   = (rho_l - rho_g ) / (rho_l + rho_g);
    double Ca   = mu_l * ulb / Sigma;
    double Pe   = ulb * static_cast<double>(lref) / M;
    double dx    = 1. / static_cast<double>(lref);
    double dt    = dx * ulb;
    return make_tuple(Re, At, Ca, Pe, dx,dt);
}

inline void printParameters_layered2D_Fakhari(const Dim_layered2D_Fakhari &dim,
                                              double Re, double At, double Ca, double Pe, double ulb,
                                              int N, double max_t)
{
    cout << "Layered 2-D (Conservative PF — BGK/BGK, paper-faithful)\n"
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
inline auto restartClock_layered2D_Fakhari()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_layered2D_Fakhari(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    int dure = us.count();
        cout << "Runtime: " << dure/1000000. << " seconds " << endl;
        cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

// ─────────────────────────   LBM functor   ─────────────────────────────────
struct LBM_layered2D_Fakhari
{
    using  CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem
                                                     + 2 * 9 * nelem; }

    // raw buffers
    CellData*                         lattice;
    CellType_layered2D_Fakhari* flag;
    int*                        parity;
    std::array<int,2>*          c;
    int*                        opp;
    double*                     t;

    // stored macroscopic velocity (we need to stre the velocity to use it in eq. 30)
    //double* Ux;
    //double* Uy;

    // model params
    double phi_l;
    double phi_g;         // 1 and 0
    double rho_l;
    double rho_g;         // heavy (l), light (g)
    double mu_l;        //viscosity of liquid
    double mu_g;        // visosity of gas
    double Sigma;
    double Fbx;              // Body force in x
    double Fby;              // Body force in y
    double xi;
    double M;
    Dim_layered2D_Fakhari dim;

    // constants
    static constexpr double cs2 = 1.0/3.0;

    // ───────── index helpers ─────────
    size_t xyz_to_i (int x, int y) {
        return (y + dim.ny * x);
    };

    auto i_to_xyz (int i) {
        int iX = i / (dim.ny);
        int remainder = i % (dim.ny);
        int iY = remainder;
        return std::make_tuple(iX, iY);
    };

    // ───────── array accessors (ping-pong) ───────
    double& f   (int i,int k)  { return lattice[              k*dim.nelem + i]; }
    double& fin (int i,int k)  { return lattice[*parity  *dim.npop + k*dim.nelem + i]; }
    double& fout(int i,int k)  { return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }

    double& g   (int i,int k)  { return lattice[2*dim.npop                  + k*dim.nelem + i]; }
    double& gin (int i,int k)  { return lattice[*parity  *dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    double& gout(int i,int k)  { return lattice[(1-*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    // ───────────────────────── initialization (two-pass; uses precomputed φ₀, ∇φ₀) ──────────────────────────────
    /*void iniLattice(double& f0)
    {
        auto i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz(static_cast<int>(i));

        // --- layered geometry (horizontal interface) ---
        double yc = 0.5 * double(dim.ny);   // domain center y (cell-centered coords)

        // cell-center coordinates
        double X = double(iX) + 0.5;
        double Y = double(iY) + 0.5;

        // smooth tanh interface across Y: upper half -> phi_l, lower half -> phi_g
        double phi0 = 0.5*(phi_l + phi_g);
        double amp  = 0.5*(phi_l - phi_g);
        double phi  = phi0 + amp * std::tanh( (Y - yc) / (0.5*xi) );

        // populate both distribution sets with equilibria at U=0
        for (int k = 0; k < 9; ++k)
        {
            fin(i,k) = phi * t[k];
            gin(i,k) = 0.;
        }
    };*/

    void iniLattice(double& f0)
    {
        auto i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz(static_cast<int>(i));

        // Fraction of domain occupied by the liquid slab
        const double slab_frac = 0.4;   // 40% of the height
        const double xi = 2.0;          // interface thickness (cells)

        const double H = double(dim.ny);

        // Cell-center coordinate
        const double Y = double(iY) + 0.5;

        // Slab centered vertically
        const double yc = 0.5 * H;
        const double half_thickness = 0.5 * slab_frac * H;

        const double y_low  = yc - half_thickness;
        const double y_high = yc + half_thickness;

        // Smooth indicator: ~1 inside slab, ~0 outside
        const double s_low  = 0.5 * (1.0 + std::tanh((Y - y_low ) / xi));
        const double s_high = 0.5 * (1.0 - std::tanh((Y - y_high) / xi));
        const double s_mid  = s_low * s_high;

        // Liquid in middle, gas outside
        const double phi = s_mid * phi_l + (1.0 - s_mid) * phi_g;

        // Initialize equilibrium distributions at rest
        for (int k = 0; k < 9; ++k)
        {
            fin(i, k) = phi * t[k];
            gin(i, k) = 0.0;
        }
    }


// ───────────────────────── kernels ─────────────────────────

    // macro-vars ϕ, P, u
    auto macro_phi_P(double& f0){   // Eq. 12 , Eq. 32a
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

        //if (phi < 1e-10) phi = 1e-30;
        return make_tuple(phi, P);
    }

    auto macro_u(double& f0){   // Eq. 32b first term
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

    double local_rho(double& f0){   // Eq. 13
        int i = &f0 - lattice;

        auto [phi, P] = macro_phi_P(f0);

        //return rho_g + ((phi - phi_g)/(phi_l - phi_g)) * (rho_l - rho_g);   // can be use of both cases: Phi_g = 0, phi_l = 1 and phi_g = -0.5, phi_l = 0.5
        return rho_g + (phi - phi_g) * (rho_l - rho_g);
    }


    double local_mu(double& f0){    // Eq. 24
        int i = &f0 - lattice;

        auto [phi , p] = macro_phi_P(f0);

        //return mu_g + ((phi - phi_g)/(phi_l - phi_g)) * (mu_l - mu_g);   // can be use of both cases: Phi_g = 0, phi_l = 1 and phi_g = -0.5, phi_l = 0.5
        return mu_g + (phi - phi_g) * (mu_l - mu_g);
    }


    double local_tau(double& f0){   // Eq. 25
        int i = &f0 - lattice;

        auto rho = local_rho(f0);
        auto mu = local_mu(f0);
        return mu / (rho * cs2);
    }


    double local_nu(double& f0){    // Eq. 21
        int i = &f0 - lattice;

        auto tau = local_tau(f0);
        return tau * cs2;
    }

    double mu_phi(double& f0){    // Eq. 5
        int i = &f0 - lattice;

        auto [phi , p] = macro_phi_P(f0);
        double beta = 12.0 * Sigma / xi;
        double kappa = 3. * Sigma * xi / 2.;

        double phi0 = (phi_l + phi_g)/2.;

        double lap_phi = laplacian_phi(f0);

        return 4 * beta * (phi - phi_g) * ( phi - phi_l) * ( phi - phi0) - kappa * lap_phi;

    }
    // ------------------------------------------------------------------
    // ∇phi   (lattice Gradient of the local density phi)
    // ------------------------------------------------------------------
    std::array<double,2> gradient_phi(double& f0){  // Eq. 34
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(int(i));

        double gx=0.0, gy=0.0;
        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_layered2D_Fakhari::bounce_back) {  // TODO : how to calculate gradient at the boundaries?
                int ixbb = iX - c[k][0];
                int iybb = iY - c[k][1];
                ixbb = (ixbb + dim.nx) % dim.nx;
                int nbb = xyz_to_i(ixbb,iybb);
                auto [phi_nb, P_nb] = macro_phi_P(lattice[nbb]);
                gx += t[k] * c[k][0] * phi_nb;
                gy += t[k] * c[k][1] * phi_nb;
            }
            else {
                auto [phi_nbb, P_nbb] = macro_phi_P(lattice[nb]);
                gx += t[k] * c[k][0] * phi_nbb;
                gy += t[k] * c[k][1] * phi_nbb;
            }
        }
        return { gx / cs2, gy / cs2};
    }


    // ------------------------------------------------------------------
    // ∇²phi   (lattice Laplacian of the local density phi)
    // ------------------------------------------------------------------
    double laplacian_phi(double& f0){   // Eq. 35
        size_t i   = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(i);

        auto [phi_c, p] = macro_phi_P(lattice[i]);

        double sum   = 0.0;

        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            
            int nb = xyz_to_i(ix,iy);

            if (flag[nb] != CellType_layered2D_Fakhari::bounce_back) {  // TODO : how to calculate laplacian at the boundaries?
                auto [phi , P] = macro_phi_P(lattice[nb]);
                sum += t[k] * (phi - phi_c);
            }
        }
        return 2. * sum / cs2;
    }


    // ------------------------------------------------------------------
    // ∇ρ   (lattice Gradient of the local density rho)
    // ------------------------------------------------------------------
    std::array<double,2> gradient_rho(double& f0){  // Eq. 33 , but the mathematical gradient is also written.
        size_t i     = &f0 - lattice;

        /* double gx=0.0, gy=0.0;
        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            iy = (iy + dim.ny) % dim.ny;
            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_layered2D_Fakhari::bounce_back) {  // TODO : how to calculate gradient at the boundaries?
                int ixbb = iX - c[k][0];
                int iybb = iY - c[k][1];
                ixbb = (ixbb + dim.nx) % dim.nx;
                iybb = (iybb + dim.ny) % dim.ny;
                int nbb = xyz_to_i(ixbb,iybb);
                double rho = local_rho(lattice[nbb]);
                gx += t[k] * c[k][0] * rho;
                gy += t[k] * c[k][1] * rho;
            }
            else {
                double rho = local_rho(lattice[nb]);
                gx += t[k] * c[k][0] * rho;
                gy += t[k] * c[k][1] * rho;
            }
        }
        return { gx / CS2, gy / CS2}; */

        auto grad_phi = gradient_phi(f0);
        double delta_rho = (rho_l - rho_g);

        return { delta_rho * grad_phi[0], delta_rho * grad_phi[1]};
    }





    // ------------------------------------------------------------------
    // Forces (Fs (Eq. 4) , Fb (not mentioned in the article, body force) , Fp (Eq. 19) , Fmu (Eq. 30 ( for BGK)))
    // ------------------------------------------------------------------
    std::array<double,2> Fs(double& f0){
        size_t i     = &f0 - lattice;

        auto grad_phi = gradient_phi(f0);
        double mu = mu_phi(f0);

        return { mu * grad_phi[0], mu * grad_phi[1]};
    }

    std::array<double,2> Fb(double& f0){
        size_t i     = &f0 - lattice;

        double rho = local_rho(f0);

        return { rho * Fbx, rho * Fby};
    }


    std::array<double,2> Fp(double& f0){
        size_t i     = &f0 - lattice;

        auto [phi , Pstar] = macro_phi_P(f0);
        auto grad_rho = gradient_rho(f0);

        return { -Pstar * cs2 * grad_rho[0], -Pstar * cs2 * grad_rho[1]};
    }


    std::array<double,2> Fmu(double& f0){
        auto i = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(i);

        auto [phi, pstar] = macro_phi_P(f0);
        auto u            = mid_velocity(f0);
        double rho  = local_rho(f0);
        double tau  = local_tau(f0);
        double nu   = local_nu(f0);
        double fac  = - nu / ((tau + 0.5) * cs2);

        auto grad_rho = gradient_rho(f0);

        double usqr = (u[0]*u[0] + u[1]*u[1]) / (2.0*cs2);

        double Sxx = 0.0, Sxy = 0.0, Syy = 0.0;
        for(int k=0; k<9; ++k){
            if (k==4) continue;

            double ex = c[k][0];
            double ey = c[k][1];
            double ck_u = ex*u[0] + ey*u[1];

            double gamma_k = t[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - usqr);
            double geq_k   = pstar * t[k] + (gamma_k - t[k]);

            double gneq = gin(i,k) - geq_k;

            Sxx += gneq * ex * ex;
            Sxy += gneq * ex * ey;
            Syy += gneq * ey * ey;
        }

        double Fx = fac * ( Sxx * grad_rho[0] + Sxy * grad_rho[1] );
        double Fy = fac * ( Sxy * grad_rho[0] + Syy * grad_rho[1] );
        return {Fx, Fy};
    }


    std::array<double,2> Total_Fsbp(double& f0){
        size_t i     = &f0 - lattice;

        auto F_s = Fs(f0);
        auto F_b = Fb(f0);
        auto F_p = Fp(f0);
        //auto F_mu = Fmu(f0);

        return {F_s[0] + F_b[0] + F_p[0] , F_s[1] + F_b[1] + F_p[1]};
    }

    std::array<double,2> Total_F(double& f0){
        size_t i     = &f0 - lattice;

        auto F_s = Fs(f0);
        auto F_b = Fb(f0);
        auto F_p = Fp(f0);
        auto F_mu = Fmu(f0);

        return {F_s[0] + F_b[0] + F_p[0] + F_mu[0] , F_s[1] + F_b[1] + F_p[1] + F_mu[1]};
    }


    // ------------------------------------------------------------------
    // Velocity
    // ------------------------------------------------------------------

    std::array<double,2> mid_velocity(double& f0){  // Eq. 32b
        size_t i     = &f0 - lattice;

        auto u = macro_u(f0);
        auto force = Total_Fsbp(f0);
        double rho = local_rho(f0);

        return{u[0] + force[0]/(2.*rho), u[1] + force[1]/(2.*rho)};
    }

    std::array<double,2> velocity(double& f0){  // Eq. 32b
        size_t i     = &f0 - lattice;

        auto u = macro_u(f0);
        auto force = Total_F(f0);
        double rho = local_rho(f0);

        return{u[0] + force[0]/(2.*rho), u[1] + force[1]/(2.*rho)};
    }

    std::array<double,2> Fphi_fac(double& f0){  // Eq. 7 (only the k-independent part)
        size_t i     = &f0 - lattice;

        auto [phi , p] = macro_phi_P(f0);
        auto grad_phi = gradient_phi(f0);

        double phi0 = (phi_g + phi_l) / 2.;

        double fac = (1 - 4*(phi - phi0)*(phi - phi0)) / xi;

        double s = sqrt(grad_phi[0] * grad_phi[0] + grad_phi[1] * grad_phi[1]) + 1e-30;


        return{fac * grad_phi[0] / s, fac * grad_phi[1] / s};
    }



    // single-population stream
    void stream(int i,int k,int iX,int iY,double pf,double pg)
    {
        int XX = iX + c[k][0];
        int YY = iY + c[k][1];
        XX = (XX + dim.nx) % dim.nx;

        size_t nb = xyz_to_i(XX,YY);

        if (flag[nb] == CellType_layered2D_Fakhari::bounce_back) {
            fout(i,opp[k]) = pf;
            gout(i,opp[k]) = pg;
        }
        else {
            fout(nb,k) = pf;
            gout(nb,k) = pg;
        }
    };

    // BGK collision for the pair (f,g)
    std::tuple<double,double,double,double>
    collideBgk(int i,int k,const std::array<double,2>& u,double rho,
               double phi, double P, double usqr, double tau,
               std::array<double,2>& grad_phi,
               double lap_phi,
               std::array<double,2>& grad_rho,
               std::array<double,2>& F,
               std::array<double,2>& F_phi){

        double ck_u = c[k][0]*u[0] + c[k][1]*u[1];

        double tau_phi = M / cs2;

        double gamma = t[k] * (1 + ck_u / cs2 + ck_u*ck_u / (2 * cs2 * cs2) - usqr);  // Eq. 10
        double gamma_opp = t[k] * (1 - ck_u / cs2 + ck_u*ck_u / (2 * cs2 * cs2) - usqr);

        double geq = P * t[k] + gamma - t[k];   // Eq. 17
        double geq_opp = P * t[k] + gamma_opp - t[k];

        double F_f = t[k] * (F_phi[0] * c[k][0] + F_phi[1] * c[k][1]);

        double feq_bar = phi * gamma - F_f / 2.;   // Eq. 9
        double feq_bar_opp = phi * gamma_opp + F_f / 2.;

        double pf = fin(i,k) - (fin(i,k) - feq_bar) / (0.5 + tau_phi) + F_f;  // Eq. 6
        double pf_opp = fin(i,opp[k]) - (fin(i,opp[k]) - feq_bar_opp) / (0.5 + tau_phi) - F_f;

        auto F_g = /*(1 - 0.5 / tau)*/ t[k] * (F[0] * c[k][0] + F[1] * c[k][1]) / (rho * cs2);  // Eq. 15 ( explained more in the paragraph under Eq. 17)

        double geq_bar = geq - F_g/2.;  // Eq. 16
        double geq_bar_opp = geq_opp + F_g/2.;


        double pg = gin(i,k) - (gin(i,k) - geq_bar) / (0.5 + tau) + F_g;  // Eq. 6
        double pg_opp = gin(i,opp[k]) - (gin(i,opp[k]) - geq_bar_opp) / (0.5 + tau) - F_g;

        return {pf, pg, pf_opp, pg_opp};
    }

    // main per-cell operator
    void operator()(double& f0)
    {
        int i = &f0 - lattice;
        if (flag[i] == CellType_layered2D_Fakhari::bulk) {
            auto  u   = velocity(f0);
            auto [phi, Pstar] = macro_phi_P(f0);
            double rho = local_rho(f0);

            auto grad_rho = gradient_rho(f0);
            auto grad_phi = gradient_phi(f0);
            auto lap_phi = laplacian_phi(f0);

            auto tau = local_tau(f0);

            auto F = Total_F(f0);
            auto F_phi = Fphi_fac(f0);

            double usqr = (u[0]*u[0] + u[1]*u[1]) / ( 2. * cs2 );

            auto [iX,iY] = i_to_xyz(i);

            for (int k = 0; k < 4; ++k) {
                auto [pf, pg, pf_op, pg_op] = collideBgk(
                    i, k,
                    u, rho, phi, Pstar, usqr, tau,
                    grad_phi, lap_phi, grad_rho,
                    F, F_phi
                );
                stream(i,     k   , iX, iY, pf   , pg);
                stream(i, opp[k], iX, iY, pf_op, pg_op);
            }

            // rest population (k = 4)
            {
                int k = 4;
                double tau_phi = M / cs2;

                double feq0 = phi * t[k] * (1. - usqr);
                double geq0 = t[k]*(Pstar - usqr);

                auto F_g = /*(1 - 0.5 / tau)*/ t[k] * (F[0] * c[k][0] + F[1] * c[k][1]) / (rho * cs2);  // Eq. 15 ( explained more in the paragraph under Eq. 17)
                //double feq_bar0 =  geq0                       // F_phi = 0 for k = 4 in this scheme :)
                double geq_bar0 = geq0 - 0.5 * F_g;



                fout(i,k) = fin(i,k) - (fin(i,k) - feq0) / (tau_phi + 0.5);
                gout(i,k) = gin(i,k) - (gin(i,k) - geq_bar0) / (tau + 0.5) + F_g;
            }
        }
    }
};


// ───────────────────────── VTK output ─────────────────────────────────
void saveVtkFields_layered2D_Fakhari(LBM_layered2D_Fakhari& lbm, int time_iter, double dx = 0.) {
    Dim_layered2D_Fakhari const& dim = lbm.dim;
    if (dx == 0.0) dx = 1.0 / dim.ny;
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
            float val = 0.;
            if (lbm.flag[i] == CellType_layered2D_Fakhari::bulk) {
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
            float val = 0.;
            if (lbm.flag[i] == CellType_layered2D_Fakhari::bulk) {
                val = lbm.local_rho(lbm.lattice[i]);
            }
            os << val << " ";
        }
        os << "\n";
    }
    os << "\n";

    // pstar_times_rho_cs2
    os << "SCALARS pressure float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double val = 0.;
            if (lbm.flag[i] == CellType_layered2D_Fakhari::bulk) {
                auto [phi, p] = lbm.macro_phi_P(lbm.lattice[i]);
                double rho = lbm.local_rho(lbm.lattice[i]);
                val = p * rho /3.;            }
            os << val << " ";
        }
        os << "\n";
    }
    os << "\n";

    // Velocity (vector field) — diagnostic (uses gathered forces)
    os << "VECTORS velocity float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            double ux = 0., uy = 0.;
            if (lbm.flag[i] == CellType_layered2D_Fakhari::bulk) {
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
            os << ((lbm.flag[i] == CellType_layered2D_Fakhari::bounce_back) ? 1 : 0) << " ";
        }
        os << "\n";
    }
    os << "\n";
}

// ───────────────────────── Energy computation (use stored Ux,Uy) ──────────────────────────────
double computeEnergy_layered2D_Fakhari(LBM_layered2D_Fakhari& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_layered2D_Fakhari::bulk) {
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
void inigeom_layered2D_Fakhari(LBM_layered2D_Fakhari& lbm) {
    Dim_layered2D_Fakhari const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (y == dim.ny - 1 || y == 0) {
                lbm.flag[i] = CellType_layered2D_Fakhari::bounce_back;
                for (int k = 0; k < 9; ++k) {
                    lbm.fin(i,k) = 0.0;
                    lbm.gin(i,k) = 0.0;
                }
            }
            else {
                lbm.flag[i] = CellType_layered2D_Fakhari::bulk;
            }
        }
    }
}

// ─────────────────────────   main driver (Layered)  ────────────────────────
void layered2D_Fakhari()
{
    ifstream contfile("../apps/Config_Files/config_layered2D_Fakhari.txt");
    if (!contfile.is_open()) {
        throw invalid_argument(
            "Config file not found. It should be named "
            "\"config_layered2D_Fakhari.txt\" in apps/Config_Files.");
    }

    // Parameters (match standalone .cpp)
    double ulb=1.0, max_t=0.01, phi_l=1.0, phi_g=0.0, rho_l=1.0, rho_g=1e-3;
    double Sigma=0.01, xi=4.0, M=0.02, R=0.0, Fbx=0, Fby = 0;
    double mu_l=0.1, mu_g=0.0001;
    int N=128, out_freq=100, vtk_freq=100; // heavy drop inside light, rectangle

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
    Dim_layered2D_Fakhari dim {10, N};
    auto [Re, At, Ca, Pe, dx,dt] = lbParameters_layered2D_Fakhari(ulb, N, mu_l, Sigma, rho_l , rho_g , M);
    printParameters_layered2D_Fakhari(dim, Re, At, Ca, Pe, ulb, N, max_t);

    // allocate distributions
    vector<CellData> lattice_vect(LBM_layered2D_Fakhari::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();

    // flags
    vector<CellType_layered2D_Fakhari> flag_vect(dim.nelem);
    CellType_layered2D_Fakhari* flag = &flag_vect[0];

    // parity
    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];

    // velocity storage (delayed)
    //vector<double> Ux(dim.nelem, 0.0), Uy(dim.nelem, 0.0);

    auto [c_vect, opp_vect, t_vect] = d2q9_constants_layered2D_Fakhari();

    LBM_layered2D_Fakhari lbm{
        lattice, flag, parity,
        c_vect.data(), opp_vect.data(), t_vect.data(),
        /*Ux.data(), Uy.data(),*/
        phi_l, phi_g, rho_l, rho_g,
        mu_l, mu_g, Sigma,
        Fbx, Fby, xi , M,
        dim
    };



    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });

    inigeom_layered2D_Fakhari(lbm);

    auto [start, clock_iter] = restartClock_layered2D_Fakhari();
    ofstream efile("energy.dat");

    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
            saveVtkFields_layered2D_Fakhari(lbm, time_iter);
        }
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(4) << time_iter * dt << setprecision(3)
                 << " [" << time_iter * dt / max_t * 100. << "%]" << endl;
            double energy = computeEnergy_layered2D_Fakhari(lbm) * dx * dx / (dt * dt);
            cout << "Average energy: " << setprecision(8) << energy << endl;
            efile << setw(10) << time_iter * dt << setw(16) << setprecision(8) << energy << endl;
        }
        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);

        // swap buffers
        *parity = 1 - *parity;
        ++clock_iter;
    }

    printMlups_layered2D_Fakhari(start, clock_iter, dim.nelem);
}
