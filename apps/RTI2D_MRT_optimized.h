//
// RTI2D_MRT_optimized.h
//
// This header provides an implementation of the two‑dimensional
// Rayleigh–Taylor instability (RTI) model using a conservative
// phase‑field formulation.  The hydrodynamic populations are evolved
// with a multiple‑relaxation‑time (MRT) collision operator in moment
// space, while the phase‑field populations remain BGK (single
// relaxation) as in the original scheme.  The implementation closely
// follows the structure of the BGK code but replaces the BGK
// momentum update with a MRT formulation.  The transformation
// matrices used here correspond to the conventional D2Q9 ordering;
// lookup tables are provided to map between the custom ordering used
// in this code and the standard ordering assumed by the MRT matrices.

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
#include <tuple>
#include <stdexcept>
#include <cstdlib>
#include <numeric>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>

using namespace std;
using namespace std::chrono;

// ─────────────────────────   cell‑level helpers   ───────────────────────────
// Enum identifying lattice nodes: bounce‑back or bulk.  Bounce‑back
// nodes enforce no‑slip boundary conditions; bulk nodes take part in
// the full collision–streaming process.
//enum class CellType_RTI2D_MRT : uint8_t { bounce_back, bulk };
enum class CellType_RTI2D_MRT : uint8_t { bounce_back, free_slip, bulk };
using  CellData = double;

// ─────────────────────────   D2Q9 constants   ───────────────────────────────
// The D2Q9 velocities, opposite directions and weights in the custom
// ordering used by this implementation.  The velocity set is
// enumerated beginning at the negative x axis (k=0) and proceeding
// clockwise, with k=4 corresponding to the rest population.  This
// ordering differs from the conventional D2Q9 ordering; lookup tables
// defined below convert between the two.
inline auto d2q9_constants()
{
    // Custom ordering: e0:(-1,0), e1:(-1,1), e2:(0,1), e3:(1,1), e4:(0,0),
    // e5:(1,0), e6:(1,‑1), e7:(0,-1), e8:(‑1,-1)
    vector<array<int, 2>> c_vect = {                    
        {-1, 0}, { 0,-1},           
        {-1,-1}, {-1, 1}, 
        { 0, 0},
        { 1, 0}, { 0, 1}, 
        { 1, 1}, { 1,-1}, 
    };

    // Opposite directions in custom ordering
    vector<int> opp_vect = {
        5, 6, 7, 8, 4,
        0, 1, 2, 3
    };

    vector<int> mirrorx_vect = {
        5, 1, 7, 8, 4,
        0, 6, 2, 3
    };

    vector<int> mirrory_vect = {
        0, 6, 3, 2, 4,
        5, 1, 8, 7
    };

    // Weights
    vector<double> w_vect = {
        1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 4.0/9.0,
        1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0
    };
    return make_tuple(c_vect, opp_vect, w_vect);
}


// ─────────────────────────   geometry helpers   ────────────────────────────
// Holds the lattice dimensions and derived counts.  For a 2D domain
// of size nx×ny, nelem is the number of lattice nodes and npop is the
// total number of populations (9 per node for the two sets of
// distribution functions, f and g).
struct Dim_RTI2D_MRT {
    Dim_RTI2D_MRT(int nx_, int ny_)
    : nx(nx_), ny(ny_),
      nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)),
      npop (9 * nelem) {}
    int    nx, ny;
    size_t nelem, npop;
};

// ─────────────────────────   lattice‑unit parameters   ─────────────────────
// Convert physical parameters into lattice units.  Returns the
// Reynolds, Atwood, Capillary and Peclet numbers together with the
// lattice spacing (dx) and time step (dt).
inline auto lbParameters_RTI2D_MRT(double ulb, int lref, double mu_l, double Sigma,
                                   double rho_l, double rho_g, double M)
{
    double Re   = rho_l * ulb * static_cast<double>(lref) / mu_l;
    double At   = (rho_l - rho_g) / (rho_l + rho_g);
    double Ca   = mu_l * ulb / Sigma;
    double Pe   = ulb * static_cast<double>(lref) / M;
    double dx   = 1.0 / static_cast<double>(lref);
    double dt   = dx * ulb;
    return make_tuple(Re, At, Ca, Pe, dx, dt);
}

// Pretty‑print the problem parameters.
inline void printParameters_RTI2D_MRT(const Dim_RTI2D_MRT &dim,
                                      double Re, double At, double Ca,
                                      double Pe, double ulb,
                                      int N, double max_t)
{
    cout << "Rayleigh–Taylor 2‑D (Conservative PF — MRT/MRT)\n"
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
inline auto restartClock_RTI2D_MRT()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_RTI2D_MRT(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

// ─────────────────────────   LBM functor   ─────────────────────────────────
// The core of the MRT implementation.  Each instance holds the state
// and parameters for a single RTI simulation.  The operator() is
// invoked in parallel over all lattice nodes to perform collision and
// streaming.
struct LBM_RTI2D_MRT
{
    // total number of entries in the distribution array
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem + 2 * 9 * nelem; }

    using CellData = double;
    static constexpr int    Q   = 9;
    static constexpr double cs2 = 1.0/3.0;
    static constexpr double PI  = 3.1415926535897932384626433832795;

    // raw buffers
    CellData* lattice;
    CellType_RTI2D_MRT* flag;
    int* parity;
    std::array<int,2>* c;   // e_k
    int* opp;
    double* w;              // weights
    int* nb    = nullptr;
    int* nbb   = nullptr;
    int* nbmx  = nullptr;   // mirrored neighbor for vertical free-slip walls
    int* nbmy  = nullptr;   // mirrored neighbor for horizontal free-slip walls

    // model parameters
    double phi_l, phi_g;
    double rho_l, rho_g;
    double mu_l,  mu_g;
    double Sigma;
    double Fbx, Fby;
    double xi;
    double M;               // mobility
    Dim_RTI2D_MRT dim;

    // MRT transformation matrices and lookup tables.  These are
    // declared constexpr so they can be used at compile time.  The
    // notation follows the description in the user request: MRT_M is
    // the forward transform from velocity space to moment space and
    // MRT_M_inv is its inverse.  custom_to_std maps from the custom
    // ordering (this code) to the standard D2Q9 ordering assumed by
    // the MRT matrices, while std_to_custom performs the inverse
    // mapping.
    static constexpr double MRT_M[9][9] = {
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

    static constexpr double MRT_M_inv[9][9] = {
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

    static constexpr int custom_to_std[9] = {3, 4, 7, 6, 0, 1, 2, 5, 8};
    static constexpr int std_to_custom[9]  = {4, 5, 6, 0, 1, 7, 3, 2, 8};

    // index helpers: convert between (x,y) and linear index i
    inline size_t xyz_to_i(int x,int y) const { return static_cast<size_t>(y + dim.ny * x); }
    inline auto i_to_xyz(int i) const {
        int iX = i / dim.ny;
        int iY = i % dim.ny;
        return std::make_tuple(iX,iY);
    }

    inline int nearest_bulk_for_wall(int i) const
    {
        auto [x, y] = i_to_xyz(i);

        int xb = std::clamp(x, 1, dim.nx - 2);
        int yb = std::clamp(y, 1, dim.ny - 2);

        return static_cast<int>(xyz_to_i(xb, yb));
    }

    inline double phi_for_output(int i) const
    {
        if (flag[i] == CellType_RTI2D_MRT::bulk)
            return macro_phi_only(i);

        int j = nearest_bulk_for_wall(i);
        return macro_phi_only(j);
    }

    // population access: layout = 4 * Q * nelem (f,fin,g,gin ping‑pong)
    inline double& fin (int i,int k)        { return lattice[*parity      *dim.npop + k*dim.nelem + i]; }
    inline double& fout(int i,int k)        { return lattice[(1-*parity)  *dim.npop + k*dim.nelem + i]; }
    inline double& gin (int i,int k)        { return lattice[*parity      *dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    inline double& gout(int i,int k)        { return lattice[(1-*parity)  *dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    inline double  fin (int i,int k)  const { return lattice[*parity      *dim.npop + k*dim.nelem + i]; }
    inline double  gin (int i,int k)  const { return lattice[*parity      *dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    inline int idx_clamped(int x, int y) const
    {
        x = std::clamp(x, 0, dim.nx - 1);
        y = std::clamp(y, 0, dim.ny - 1);
        return static_cast<int>(xyz_to_i(x, y));
    }

    inline void build_neighbors()
    {
        for (int x = 0; x < dim.nx; ++x) {
            for (int y = 0; y < dim.ny; ++y) {
                int i = static_cast<int>(xyz_to_i(x, y));
                int base = i * Q;

                for (int k = 0; k < Q; ++k) {
                    // direct neighbor
                    nb[base + k] = idx_clamped(
                        x + c[k][0],
                        y + c[k][1]
                    );

                    // full opposite (bounce-back style)
                    nbb[base + k] = idx_clamped(
                        x + c[opp[k]][0],
                        y + c[opp[k]][1]
                    );

                    // mirror in x (vertical-wall free-slip)
                    nbmx[base + k] = idx_clamped(
                        x + c[mirror_x[k]][0],
                        y + c[mirror_x[k]][1]
                    );

                    // mirror in y (horizontal-wall free-slip)
                    nbmy[base + k] = idx_clamped(
                        x + c[mirror_y[k]][0],
                        y + c[mirror_y[k]][1]
                    );
                }
            }
        }
    }
    // Specular reflection maps for the custom D2Q9 ordering:
    // 0(-1,0), 1(0,-1), 2(-1,-1), 3(-1,1), 4(0,0),
    // 5( 1,0), 6(0, 1), 7( 1,1), 8( 1,-1)

    // reflect on a vertical wall: reverse x, keep y
    static constexpr int mirror_x[9] = {
        5, 1, 8, 7, 4, 0, 6, 3, 2
    };

    // reflect on a horizontal wall: reverse y, keep x
    static constexpr int mirror_y[9] = {
        0, 6, 3, 2, 4, 5, 1, 8, 7
    };

    inline bool is_wall(int i) const {
        return flag[i] != CellType_RTI2D_MRT::bulk;
    }

    inline int slip_reflect_dir(int wall_i, int k) const
    {
        auto [x, y] = i_to_xyz(wall_i);

        const bool left   = (x == 0);
        const bool right  = (x == dim.nx - 1);
        const bool bottom = (y == 0);
        const bool top    = (y == dim.ny - 1);

        // At a corner, reflect both components
        if ((left || right) && (bottom || top)) return opp[k];

        // Vertical wall
        if (left || right) return mirror_x[k];

        // Horizontal wall
        if (bottom || top) return mirror_y[k];

        // Fallback
        return opp[k];
    }

    inline void iniLattice(double& f0)
    {
        int i = static_cast<int>(&f0 - lattice);
        auto [iX,iY] = i_to_xyz(i);

        double phi0 = 0.5 * (phi_l + phi_g);
        double amp  = 0.5 * (phi_l - phi_g);
        double Ly   = static_cast<double>(dim.ny);

        // Bulk width only: walls are at x=0 and x=nx-1
        const double Lx_int  = static_cast<double>(std::max(1, dim.nx - 2));
        const double Lx_full = 2.0 * Lx_int;   // recover original full-width scale
        const double amplitude = 0.1 * Lx_full;

        // map interior x=[1, nx-2] to [0,1]
        double x_rel = 0.0;
        if (dim.nx > 3) {
            x_rel = static_cast<double>(iX - 1) / static_cast<double>(dim.nx - 3);
            x_rel = std::clamp(x_rel, 0.0, 1.0);
        }

        double ycell = static_cast<double>(iY) + 0.5;

        // half-wave: crest attached to the left wall side of the fluid domain
        double y_interface = 0.5 * Ly + amplitude * std::cos(PI * x_rel);

        double phi = phi0 + amp * std::tanh((ycell - y_interface) / (0.5 * xi));

        for (int k = 0; k < Q; ++k) {
            fin(i,k) = phi * w[k];
            gin(i,k) = 0.0;
        }
    }

    // macroscopic fields
    inline auto macro_phi_P(int i) const {
        double phi = 0.0, P = 0.0;
        for (int k=0; k<Q; ++k) {
            phi += fin(i,k);
            P   += gin(i,k);
        }
        return std::make_tuple(phi,P);
    }

    inline double macro_phi_only(int i) const {
        double phi = 0.0;
        for (int k=0; k<Q; ++k) phi += fin(i,k);
        return phi;
    }

    // velocity from g populations (before force shift)
    inline auto macro_u(int i) const {
        double ux = 0.0, uy = 0.0;
        for (int k=0; k<Q; ++k) {
            double gk = gin(i,k);
            ux    += gk * c[k][0];
            uy    += gk * c[k][1];
        }
        return std::array<double,2>{ux, uy};
    }

    inline double local_rho(double phi) const {
        return rho_g + (phi - phi_g) * (rho_l - rho_g);
    }
    inline double local_mu(double phi) const {
        return mu_g + (phi - phi_g) * (mu_l - mu_g);
    }
    inline double local_tau(double rho,double mu) const { return mu / (rho*cs2); }
    inline double local_nu (double tau) const { return tau * cs2; }
    //inline double local_M (double tau) const {return t * cs2}

    // gradient and Laplacian of φ
inline std::array<double,2> gradient_phi(int i) const {
    double gx = 0.0, gy = 0.0;
    int base = i * Q;

    for (int k = 0; k < Q; ++k) {
        int j = nb[base + k];

        if (flag[j] == CellType_RTI2D_MRT::free_slip) {
            j = nbb[base + k];
        }
        /*else if (flag[j] == CellType_RTI2D_MRT::free_slip) {
            auto [xj, yj] = i_to_xyz(j);
            if (xj == 0 || xj == dim.nx - 1 || yj == 0 || yj == dim.ny - 1 )
                j = nbmx[base + k];
            else
                j = nbmy[base + k];
        }*/

        double phi_nb = macro_phi_only(j);
        gx += w[k] * c[k][0] * phi_nb;
        gy += w[k] * c[k][1] * phi_nb;
    }

    return { gx / cs2, gy / cs2 };
}
inline double laplacian_phi(int i) const {
    double phi_c = macro_phi_only(i);
    double sum = 0.0;
    int base = i * Q;

    for (int k = 0; k < Q; ++k) {
        int j = nb[base + k];

        if (flag[j] == CellType_RTI2D_MRT::free_slip) {
            j = nbb[base + k];
        }
        /*else if (flag[j] == CellType_RTI2D_MRT:: free_slip) {
            auto [xj, yj] = i_to_xyz(j);
            if (xj == 0 || xj == dim.nx - 1 || yj == 0 || yj == dim.ny - 1 )
                j = nbmx[base + k];
            else
                j = nbmy[base + k];
        } */

        double phi_n = macro_phi_only(j);
        sum += w[k] * (phi_n - phi_c);
    }

    return 2.0 * sum / cs2;
}

    inline std::array<double,2> gradient_rho_from_gradphi( std::array<double,2>& grad_phi) const {
        double dr = rho_l - rho_g;
        return { dr*grad_phi[0], dr*grad_phi[1] };
    }

    inline double mu_phi(double phi,double lap_phi) const {
        double beta  = 12.0*Sigma/xi;
        double kappa = 3.0*Sigma*xi/2.0;
        double phi0  = 0.5*(phi_l + phi_g);
        return 4.0*beta*(phi - phi_g)*(phi - phi_l)*(phi - phi0) - kappa*lap_phi;
    }

    inline std::array<double,2> Fs_from(double mu_phi_val,const std::array<double,2>& grad_phi) const {
        return { mu_phi_val*grad_phi[0], mu_phi_val*grad_phi[1] };
    }
    inline std::array<double,2> Fb_from(double rho) const {
        return { rho*Fbx, rho*Fby };
    }
    inline std::array<double,2> Fp_from(double Pstar,const std::array<double,2>& grad_rho) const {
        return { -Pstar*cs2*grad_rho[0], -Pstar*cs2*grad_rho[1] };
    }

/*    inline std::array<double,2> Fphi_vec(double phi,const std::array<double,2>& grad_phi) const {
        double phi0 = 0.5*(phi_g + phi_l);
        double fac  = M /cs2 * (4.0*phi*(1-phi)) / xi; // (1.0 - 4.0*(phi-phi0)*(phi-phi0)) / xi;
        double s   = std::sqrt(grad_phi[0]*grad_phi[0] + grad_phi[1]*grad_phi[1]) + 1e-32;
	return { fac*grad_phi[0]/s, fac*grad_phi[1]/s };
    }
*/
   inline std::array<double,2> Fphi_vec(double phi,const std::array<double,2>& grad_phi) const {
    	double phi0 = 0.5*(phi_g + phi_l);
    	double shape = (1.0 - 4.0*(phi - phi0)*(phi - phi0)) / xi; // Eq. (7) factor
    	double s = std::sqrt(grad_phi[0]*grad_phi[0] + grad_phi[1]*grad_phi[1]) + 1e-32;
    	double nx = grad_phi[0] / s;
    	double ny = grad_phi[1] / s;
        return { shape * nx, shape * ny };
    }

    inline std::array<double,2> Total_Fsbp(const std::array<double,2>& Fs,
                                           const std::array<double,2>& Fb,
                                           const std::array<double,2>& Fp) const
    {
        return { Fs[0] + Fb[0] + Fp[0],
                 Fs[1] + Fb[1] + Fp[1] };
    }

    inline std::array<double,2> mid_velocity(const std::array<double,2>& u_macro,
                                             const std::array<double,2>& F_sbp,
                                             double rho) const
    {
        return { u_macro[0] /*+ F_sbp[0]/(2.0*rho)*/,
                 u_macro[1] /*+ F_sbp[1]/(2.0*rho)*/ };
    }

    // Viscous correction using MRT.  This replaces the BGK‑based
    // Fmu_from in the BGK code.  Given a cell index i, the mid‑step
    // velocity u, the equilibrium pressure Pstar, the relaxation time
    // tau, the kinematic viscosity nu and the gradient of density
    // grad_rho, this routine constructs the non‑equilibrium stress
    // tensor in moment space and contracts it with grad_rho to obtain
    // the viscous force.  The implementation follows the procedure
    // outlined in the user‑provided pseudocode.
    inline std::array<double,2> Fmu_from(int i,
                                         const std::array<double,2>& u,
                                         double Pstar,double tau,double nu,
                                         const std::array<double,2>& grad_rho) const
    {
        // compute u²/2c_s² once
        double u_sq = (u[0]*u[0] + u[1]*u[1]) / (2.0*cs2);

        // gather current g distributions and equilibrium in standard ordering
        double gvec_std[9];
        double geq_std[9];
        for (int k=0; k<9; ++k) {
            int s = custom_to_std[k];
            // current distribution
            double gk = gin(i,k);
            // compute equilibrium for hydrodynamic MRT (without forcing)
            double ex = static_cast<double>(c[k][0]);
            double ey = static_cast<double>(c[k][1]);
            double ck_u = ex*u[0] + ey*u[1];
            double gamma_k = w[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - u_sq);
            double geq_k   = Pstar * w[k] + (gamma_k - w[k]);
            gvec_std[s] = gk;
            geq_std[s]  = geq_k;
        }

        // form moment vector m = M*(g - g_eq)
        double m_vec[9];
        for (int r=0; r<9; ++r) {
            double sum = 0.0;
            for (int a=0; a<9; ++a) {
                sum += MRT_M[r][a] * (gvec_std[a] - geq_std[a]);
            }
            m_vec[r] = sum;
        }

        // relaxation diagonal: only shear modes (indices 7,8) depend on tau
        double s_nu = 1.0 / (tau + 0.5);
        double Sdiag[9] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, s_nu, s_nu };

        // post‑collision moments s_m = Sdiag * m
        double s_m[9];
        for (int r=0; r<9; ++r) {
            s_m[r] = Sdiag[r] * m_vec[r];
        }

        // back to velocity space: A_std = M_inv * s_m
        double A_std[9];
        for (int a=0; a<9; ++a) {
            double sum = 0.0;
            for (int r=0; r<9; ++r) {
                sum += MRT_M_inv[a][r] * s_m[r];
            }
            A_std[a] = sum;
        }

        // reorder to custom ordering
        double A_custom[9];
        for (int k=0; k<9; ++k) {
            int s = custom_to_std[k];
            A_custom[k] = A_std[s];
        }

        // contract stress tensor with gradient of density using custom c and A_custom
        double Txx = 0.0, Txy = 0.0, Tyy = 0.0;
        for (int beta=0; beta<9; ++beta) {
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

    inline std::array<double,2> velocity_from(const std::array<double,2>& u_macro,
                                              const std::array<double,2>& F_total,
                                              double rho) const
    {
        return { u_macro[0] + F_total[0]/(2.0*rho),
                 u_macro[1] + F_total[1]/(2.0*rho) };
    }


inline std::array<double,2> total_velocity(int i) const
{
    auto [phi, Pstar] = macro_phi_P(i);
    double rho  = local_rho(phi);
    double mu   = local_mu(phi);
    double tau  = local_tau(rho, mu);
    double nu   = local_nu(tau);

    auto u_macro  = macro_u(i);
    auto grad_phi = gradient_phi(i);
    double lap_phi = laplacian_phi(i);
    auto grad_rho = gradient_rho_from_gradphi(grad_phi);

    double mu_phi_val = mu_phi(phi, lap_phi);
    auto Fs   = Fs_from(mu_phi_val, grad_phi);
    auto Fb   = Fb_from(rho);
    auto Fp   = Fp_from(Pstar, grad_rho);
    auto Fsbp = Total_Fsbp(Fs, Fb, Fp);
    auto u_mid = mid_velocity(u_macro, Fsbp, rho);
    auto Fmu   = Fmu_from(i, u_mid, Pstar, tau, nu, grad_rho);

    std::array<double,2> Ftot{
        Fs[0] + Fb[0] + Fp[0] + Fmu[0],
        Fs[1] + Fb[1] + Fp[1] + Fmu[1]
    };

    return velocity_from(u_macro, Ftot, rho);
}


    // ---- MRT collision + streaming, per direction ----
    inline void operator()(double& f0)
    {
        int i = static_cast<int>(&f0 - lattice);
        // skip bounce‑back nodes: they are handled elsewhere during streaming
        if (flag[i] != CellType_RTI2D_MRT::bulk) return;

        // macroscopic quantities
        auto [phi,Pstar] = macro_phi_P(i);
        double rho  = local_rho(phi);
        double mu   = local_mu(phi);
        double tau  = local_tau(rho,mu);
        double nu   = local_nu(tau);
        auto u_macro= macro_u(i);

        // gradients and intermediate forces
        auto grad_phi = gradient_phi(i);
        double lap_phi= laplacian_phi(i);
        auto grad_rho = gradient_rho_from_gradphi(grad_phi);
        double mu_phi_val = mu_phi(phi,lap_phi);
        auto Fs   = Fs_from(mu_phi_val,grad_phi);
        auto Fb   = Fb_from(rho);
        auto Fp   = Fp_from(Pstar,grad_rho);
        auto F_sbp= Total_Fsbp(Fs,Fb,Fp);
        auto u_mid= mid_velocity(u_macro,F_sbp,rho);
        // viscous correction using MRT
        auto Fmu_vec= Fmu_from(i,u_mid,Pstar,tau,nu,grad_rho);
        // total hydrodynamic forcing
        std::array<double,2> F_tot{
            Fs[0] + Fb[0] + Fp[0] + Fmu_vec[0],
            Fs[1] + Fb[1] + Fp[1] + Fmu_vec[1]
        };
        // full velocity including forcing
        auto u = velocity_from(u_macro,F_tot,rho);
        double u_sq = (u[0]*u[0] + u[1]*u[1]) / (2.0*cs2);
        double tau_phi = M / cs2;

        // precompute Fϕ vector once per cell
        auto F_phi_vec = Fphi_vec(phi,grad_phi);

        // phase field BGK post‑collision distributions
        double pf_out[9];
        for (int k=0; k<9; ++k) {
            double ex = static_cast<double>(c[k][0]);
            double ey = static_cast<double>(c[k][1]);
            double ck_u = ex*u[0] + ey*u[1];
            // γ_k in the conservative phase‑field BGK
            double gamma_k = w[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - u_sq);
            // Fϕ term: w_k (F_phi ⋅ e_k)
            double Ff_k = w[k] * (F_phi_vec[0] * ex + F_phi_vec[1] * ey);
            double feq_bar = phi * gamma_k - 0.5 *  Ff_k ;
            double f_old   = fin(i,k);
            pf_out[k] = f_old - (f_old - feq_bar)/(tau_phi + 0.5) + Ff_k;
        }

        // hydrodynamic MRT post‑collision distributions
        double gvec_std[9];
        double geq_std[9];
        double Fg_std[9];
        for (int k=0; k<9; ++k) {
            int s = custom_to_std[k];
            double ex = static_cast<double>(c[k][0]);
            double ey = static_cast<double>(c[k][1]);
            double ck_u = ex*u[0] + ey*u[1];
            double gamma_k = w[k] * (1.0 + ck_u/cs2 + (ck_u*ck_u)/(2.0*cs2*cs2) - u_sq);
            double g_k  = gin(i,k);
            double geq_k= Pstar * w[k] + (gamma_k - w[k]);
            double F_k  = w[k] * (F_tot[0] * ex + F_tot[1] * ey) / (rho * cs2);
            gvec_std[s] = g_k;
            geq_std[s]  = geq_k;
            Fg_std[s]   = F_k;
        }
        // moment space arrays
        double m_vec[9] = {0.0}, m_eq[9] = {0.0}, Fm[9] = {0.0}, m_post[9] = {0.0};
        for (int r=0; r<9; ++r) {
            double sum = 0.0, sum_eq = 0.0, sum_f = 0.0;
            for (int a=0; a<9; ++a) {
                sum    += MRT_M[r][a] * gvec_std[a];
                sum_eq += MRT_M[r][a] * geq_std[a];
                sum_f  += MRT_M[r][a] * Fg_std[a];
            }
            m_vec[r] = sum;
            m_eq[r]  = sum_eq;
            Fm[r]    = sum_f;
        }
        // relaxation diagonal: shear modes 7 and 8 use s_nu; others use 1
        double s_nu = 1.0 / (tau + 0.5);
        double Sdiag[9] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, s_nu, s_nu };
        for (int r=0; r<9; ++r) {
            double Sr = Sdiag[r];
            m_post[r] = m_vec[r] - Sr * (m_vec[r] - m_eq[r]) + (1.0 - 0.5 * Sr) * Fm[r];
        }
        // back to velocity space
        double g_post_std[9];
        for (int a=0; a<9; ++a) {
            double sum = 0.0;
            for (int r=0; r<9; ++r) {
                sum += MRT_M_inv[a][r] * m_post[r];
            }
            g_post_std[a] = sum;
        }
        // reorder to custom ordering
        double g_post[9];
        for (int k=0; k<9; ++k) {
            int s = custom_to_std[k];
            g_post[k] = g_post_std[s];
        }

        // perform streaming: write post‑collision populations to neighbouring nodes
        for (int k=0; k<9; ++k) {
            int j = nb[i*Q + k];

            if (flag[j] == CellType_RTI2D_MRT::bounce_back) {
                // no-slip
                int kr = opp[k];
                fout(i, kr) = pf_out[k];
                gout(i, kr) = g_post[k];
            }
            else if (flag[j] == CellType_RTI2D_MRT::free_slip) {
                // free-slip / specular reflection
                int kr = slip_reflect_dir(j, k);
                fout(i, kr) = pf_out[k];
                gout(i, kr) = g_post[k];
            }
            else {
                // bulk neighbor
                fout(j, k) = pf_out[k];
                gout(j, k) = g_post[k];
            }
        }
    }
};

// ───────────────────────── VTK output ─────────────────────────────────
// Write macroscopic fields and forces to a VTK file for visualization.
inline void saveVtkFields_RTI2D_MRT(LBM_RTI2D_MRT& lbm,
                                    int time_iter, double dx = 0.0)
{
    Dim_RTI2D_MRT const& dim = lbm.dim;
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
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            float val = static_cast<float>(lbm.phi_for_output(idx));
            os << val << " ";
        }
        os << "\n";
    }
    os << "\n";


    // nu
    os << "SCALARS tau float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = (int)lbm.xyz_to_i(x, y);
            float tau = 0.f;
            if (lbm.flag[i] == CellType_RTI2D_MRT::bulk) {
                double phi = lbm.macro_phi_only(i);
                double rho = lbm.local_rho(phi);
                double mu = lbm.local_mu(phi);
                 tau = lbm.local_tau(rho , mu);
                //nu = (float)(tau / 3.);  
               // val = (float)lbm.local_rho(phi);
            }
            os << tau << " ";
        }
        os << "\n";
    }
    os << "\n";

    // ρ
    os << "SCALARS density float 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            float val = 0.f;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bulk) {
                double phi = lbm.macro_phi_only(idx);
                val = static_cast<float>(lbm.local_rho(phi));
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
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            double val = 0.0;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bulk) {
                auto [phi, p] = lbm.macro_phi_P(idx);
                double rho    = lbm.local_rho(phi);
                val = p * rho / 3.0;
            }
            os << val << " ";
        }
        os << "\n";
    }
    os << "\n";

    // Velocity (Fs + Fb + Fp, ignores viscous correction for cheap diagnostic)
    os << "VECTORS velocity float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            double ux = 0.0, uy = 0.0;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bulk) {
                auto [phi, P] = lbm.macro_phi_P(idx);
                double rho    = lbm.local_rho(phi);
                auto u_macro  = lbm.macro_u(idx);
                auto grad_phi = lbm.gradient_phi(idx);
                double lap_phi= lbm.laplacian_phi(idx);
                auto grad_rho = lbm.gradient_rho_from_gradphi(grad_phi);
                double mu_phi_val = lbm.mu_phi(phi, lap_phi);
                auto Fs   = lbm.Fs_from(mu_phi_val, grad_phi);
                auto Fb   = lbm.Fb_from(rho);
                auto Fp   = lbm.Fp_from(P, grad_rho);
                std::array<double,2> Ftot { Fs[0] + Fb[0] + Fp[0],
                                            Fs[1] + Fb[1] + Fp[1] };
                auto u = lbm.velocity_from(u_macro, Ftot, rho);
                ux = u[0];
                uy = u[1];
            }
            os << static_cast<float>(ux) << " " << static_cast<float>(uy) << " 0\n";
        }
    }
    os << "\n";

    os << "VECTORS gradphi float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            float fx = 0.f, fy = 0.f;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bulk) {
                auto grad_phi = lbm.gradient_phi(idx);
                fx = static_cast<float>(grad_phi[0]);
                fy = static_cast<float>(grad_phi[1]);
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";



    os << "VECTORS gradrho float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            float fx = 0.f, fy = 0.f;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bulk) {
		auto grad_phi = lbm.gradient_phi(idx);
                auto grad_rho = lbm.gradient_rho_from_gradphi(grad_phi);
                fx = static_cast<float>(grad_rho[0]);
                fy = static_cast<float>(grad_rho[1]);
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";


    // Individual forces and total force
    // Fs: surface tension
    os << "VECTORS Fs float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            float fx = 0.f, fy = 0.f;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bulk) {
                auto [phi, P] = lbm.macro_phi_P(idx);
                auto grad_phi = lbm.gradient_phi(idx);
                double lap_phi= lbm.laplacian_phi(idx);
                double mu_phi_val = lbm.mu_phi(phi, lap_phi);
                auto Fs = lbm.Fs_from(mu_phi_val, grad_phi);
                fx = static_cast<float>(Fs[0]);
                fy = static_cast<float>(Fs[1]);
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";

    // Fb: body force
    os << "VECTORS Fb float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            float fx = 0.f, fy = 0.f;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bulk) {
                double phi = lbm.macro_phi_only(idx);
                double rho = lbm.local_rho(phi);
                auto Fb = lbm.Fb_from(rho);
                fx = static_cast<float>(Fb[0]);
                fy = static_cast<float>(Fb[1]);
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";

    // Fp: pressure‑gradient force
    os << "VECTORS Fp float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            float fx = 0.f, fy = 0.f;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bulk) {
                auto [phi, P] = lbm.macro_phi_P(idx);
                auto grad_phi = lbm.gradient_phi(idx);
                auto grad_rho = lbm.gradient_rho_from_gradphi(grad_phi);
                auto Fp = lbm.Fp_from(P, grad_rho);
                fx = static_cast<float>(Fp[0]);
                fy = static_cast<float>(Fp[1]);
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";

    // Fmu: viscous correction force
    os << "VECTORS Fmu float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            float fx = 0.f, fy = 0.f;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bulk) {
                auto [phi, P] = lbm.macro_phi_P(idx);
                double rho    = lbm.local_rho(phi);
                double mu     = lbm.local_mu(phi);
                double tau    = lbm.local_tau(rho, mu);
                double nu     = lbm.local_nu(tau);
                auto u_macro  = lbm.macro_u(idx);
                auto grad_phi = lbm.gradient_phi(idx);
                double lap_phi= lbm.laplacian_phi(idx);
                auto grad_rho = lbm.gradient_rho_from_gradphi(grad_phi);
                double mu_phi_val = lbm.mu_phi(phi, lap_phi);
                auto Fs   = lbm.Fs_from(mu_phi_val, grad_phi);
                auto Fb   = lbm.Fb_from(rho);
                auto Fp_  = lbm.Fp_from(P, grad_rho);
                auto Fsbp = lbm.Total_Fsbp(Fs,Fb,Fp_);
                auto u_mid= lbm.mid_velocity(u_macro,Fsbp,rho);
                auto Fmu = lbm.Fmu_from(idx,u_mid,P,tau,nu,grad_rho);
                fx = static_cast<float>(Fmu[0]);
                fy = static_cast<float>(Fmu[1]);
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";

    // Ftotal: Fs + Fb + Fp + Fmu
    os << "VECTORS Ftotal float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            float fx = 0.f, fy = 0.f;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bulk) {
                auto [phi, P] = lbm.macro_phi_P(idx);
                double rho    = lbm.local_rho(phi);
                double mu     = lbm.local_mu(phi);
                double tau    = lbm.local_tau(rho, mu);
                double nu     = lbm.local_nu(tau);
                auto u_macro  = lbm.macro_u(idx);
                auto grad_phi = lbm.gradient_phi(idx);
                double lap_phi= lbm.laplacian_phi(idx);
                auto grad_rho = lbm.gradient_rho_from_gradphi(grad_phi);
                double mu_phi_val = lbm.mu_phi(phi, lap_phi);
                auto Fs   = lbm.Fs_from(mu_phi_val, grad_phi);
                auto Fb   = lbm.Fb_from(rho);
                auto Fp_  = lbm.Fp_from(P, grad_rho);
                auto Fsbp = lbm.Total_Fsbp(Fs,Fb,Fp_);
                auto u_mid= lbm.mid_velocity(u_macro,Fsbp,rho);
                auto Fmu = lbm.Fmu_from(idx,u_mid,P,tau,nu,grad_rho);
                double Fx = Fs[0] + Fb[0] + Fp_[0] + Fmu[0];
                double Fy = Fs[1] + Fb[1] + Fp_[1] + Fmu[1];
                fx = static_cast<float>(Fx);
                fy = static_cast<float>(Fy);
            }
            os << fx << " " << fy << " 0\n";
        }
    }
    os << "\n";


    os << "SCALARS Flag int 1\n";
    os << "LOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t idx = lbm.xyz_to_i(x, y);
            int f = 0;
            if (lbm.flag[idx] == CellType_RTI2D_MRT::bounce_back) f = 1;
            else if (lbm.flag[idx] == CellType_RTI2D_MRT::free_slip) f = 2;
            os << f << " ";
        }
        os << "\n";
    }
    os << "\n";
}

// ───────────────────────── Energy computation ──────────────────────────────
inline double computeKineticEnergy_RTI2D_MRT(LBM_RTI2D_MRT& lbm)
{
    const auto& dim = lbm.dim;
    double KE = 0.0;

    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = static_cast<int>(lbm.xyz_to_i(x, y));
            if (lbm.flag[i] != CellType_RTI2D_MRT::bulk) continue;

            double phi = lbm.macro_phi_only(i);
            double rho = lbm.local_rho(phi);
            auto u = lbm.total_velocity(i);

            KE += 0.5 * rho * (u[0]*u[0] + u[1]*u[1]);
        }
    }

    return KE;
}


inline double computePotentialEnergy_RTI2D_MRT(LBM_RTI2D_MRT& lbm)
{
    const auto& dim = lbm.dim;
    double PE = 0.0;

    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = static_cast<int>(lbm.xyz_to_i(x, y));
            if (lbm.flag[i] != CellType_RTI2D_MRT::bulk) continue;

            double phi = lbm.macro_phi_only(i);
            double rho = lbm.local_rho(phi);

            double xc = static_cast<double>(x) + 0.5;
            double yc = static_cast<double>(y) + 0.5;

            PE += -rho * (lbm.Fbx * xc + lbm.Fby * yc);
        }
    }

    return PE;
}


// ───────────────────────── Geometry initialization ─────────────────────────
/*inline void inigeom_RTI2D_MRT(LBM_RTI2D_MRT& lbm)
{
    Dim_RTI2D_MRT const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));
            if (y == 0 || y == dim.ny - 1 || x == 0 || x == dim.nx - 1) {
                lbm.flag[idx] = CellType_RTI2D_MRT::bounce_back;
                for (int k = 0; k < 9; ++k) {
                    lbm.fin(idx,k) = 0.0;
                    lbm.gin(idx,k) = 0.0;
                }
            } else {
                lbm.flag[idx] = CellType_RTI2D_MRT::bulk;
            }
        }
    }
} */
inline void inigeom_RTI2D_MRT(LBM_RTI2D_MRT& lbm)
{
    const Dim_RTI2D_MRT& dim = lbm.dim;

    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int idx = static_cast<int>(lbm.xyz_to_i(x, y));

            const bool on_left   = (x == 0);
            const bool on_right  = (x == dim.nx - 1);
            const bool on_bottom = (y == 0);
            const bool on_top    = (y == dim.ny - 1);

            const bool at_corner =
                (on_left || on_right) && (on_bottom || on_top);

            const bool guard_point =
                ((x == 0 || x == dim.nx - 1) &&
                 (y == 1 || y == dim.ny - 2));

            if (/*at_corner ||*/ on_bottom || on_top /*|| guard_point*/) {
                lbm.flag[idx] = CellType_RTI2D_MRT:: free_slip; //bounce_back;
            }
            else if (on_left || on_right) {
               lbm.flag[idx] = CellType_RTI2D_MRT::free_slip;
            }
            else {
                lbm.flag[idx] = CellType_RTI2D_MRT::bulk;
            }

            if (lbm.flag[idx] != CellType_RTI2D_MRT::bulk) {
                for (int k = 0; k < 9; ++k) {
                    lbm.fin(idx,k) = 0.0;
                    lbm.gin(idx,k) = 0.0;
                }
            }
        }
    }
}

// ─────────────────────────   main driver (RTI2D MRT)  ────────────────────────
inline void rti2d_MRT_optimized()
{
    // attempt to open a configuration file; fall back to legacy name if needed
    ifstream contfile("../apps/Config_Files/config_RTI2D_MRT.txt");
    if (!contfile.is_open()) {
        contfile.open("../apps/Config_Files/config_laplace2D_Fakhari.txt");
        if (!contfile.is_open()) {
            throw invalid_argument(
                "Config file not found. Expected "
                "\"config_RTI2D_MRT.txt\" (or legacy \"config_laplace2D_Fakhari.txt\") in apps/Config_Files.");
        }
    }

    // Parameters (defaults)
    double ulb=1.0, max_t=0.01, phi_l=1.0, phi_g=0.0, rho_l=1.0, rho_g=1e-3;
    double Sigma=0.01, xi=4.0, Mmob=0.02, Fbx=0.0, Fby=0.0;
    double mu_l=0.1, mu_g=0.0001;
    int N=256, out_freq=100, vtk_freq=100;

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
        else if (param == "M")          Mmob    = stod(value);
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
    //Dim_RTI2D_MRT dim {N, 4 * N+2};
    Dim_RTI2D_MRT dim {N/2, 4 * N + 2};

    auto [Re, At, Ca, Pa, dx, dt] = lbParameters_RTI2D_MRT(ulb, N, mu_l, Sigma, rho_l , rho_g, Mmob);
    printParameters_RTI2D_MRT(dim, Re, At, Ca, Pa, ulb, N, max_t);

    // allocate distributions
    vector<CellData> lattice_vect(LBM_RTI2D_MRT::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();

    // flags
    vector<CellType_RTI2D_MRT> flag_vect(dim.nelem);
    CellType_RTI2D_MRT* flag = flag_vect.data();

    // parity
    vector<int> parity_vect {0};
    int* parity = parity_vect.data();

    // neighbors
    vector<int> nb_vect(dim.nelem * 9);
    vector<int> nbb_vect(dim.nelem * 9);
    vector<int> nbmx_vect(dim.nelem * 9);
    vector<int> nbmy_vect(dim.nelem * 9);

    // constants
    auto [c_vect, opp_vect, t_vect] = d2q9_constants();

LBM_RTI2D_MRT lbm{
    lattice,
    flag,
    parity,
    c_vect.data(),
    opp_vect.data(),
    t_vect.data(),
    /* nb   */ nullptr,
    /* nbb  */ nullptr,
    /* nbmx */ nullptr,
    /* nbmy */ nullptr,
    phi_l, phi_g, rho_l, rho_g,
    mu_l, mu_g, Sigma,
    Fbx, Fby, xi, Mmob,
    dim
};

    // then bind the neighbor tables you allocated above:
    lbm.nb   = nb_vect.data();
    lbm.nbb  = nbb_vect.data();
    lbm.nbmx = nbmx_vect.data();
    lbm.nbmy = nbmy_vect.data();
    lbm.build_neighbors();


    // init fields
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });
    inigeom_RTI2D_MRT(lbm);

    auto [start, clock_iter] = restartClock_RTI2D_MRT();
    //ofstream efile("energy.dat");
    ofstream efile("energies.dat");
    efile << "# t KE PE Etotal\n";

    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
            saveVtkFields_RTI2D_MRT(lbm, time_iter);
        }
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(6) << time_iter * dt << setprecision(3)
                 << " [" << (time_iter * dt / max_t * 100.0) << "%]" << endl;

            double KE = computeKineticEnergy_RTI2D_MRT(lbm);
            double PE = computePotentialEnergy_RTI2D_MRT(lbm);
            double ET = KE + PE;

            cout << "KE = " << setprecision(8) << KE
                 << "  PE = " << PE
                 << "  E = " << ET << endl;

            efile << setw(12) << setprecision(6) << time_iter * dt
                  << setw(18) << setprecision(10) << KE
                  << setw(18) << setprecision(10) << PE
                  << setw(18) << setprecision(10) << ET << "\n";
        }
        // main step — parallelized over cells
        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        // swap buffers
        *parity = 1 - *parity;
        ++clock_iter;
    }
    printMlups_RTI2D_MRT(start, clock_iter, dim.nelem);
}
