#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <execution>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace std;
using namespace std::chrono;

enum class CellType_laplace2D_Liang : uint8_t { bounce_back, bulk };
using CellData = double;

inline auto d2q9_constants_laplace2D_Liang() {
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

struct Dim_laplace2D_Liang {
    Dim_laplace2D_Liang(int nx_, int ny_)
    : nx(nx_), ny(ny_), nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)), npop(9 * nelem) {}
    int nx, ny;
    size_t nelem, npop;
};

inline auto lbParameters_laplace2D_Liang(double ulb, int lref, double nu_l, double Sigma, double rho_l, double rho_g, double M) {
    double Re = rho_l * ulb * static_cast<double>(lref) / nu_l;
    double At = (rho_l - rho_g) / (rho_l + rho_g);
    double Ca = nu_l * rho_l * ulb / Sigma;
    double Pe = ulb * static_cast<double>(lref) / M;
    double dx = 1. / static_cast<double>(lref);
    double dt = dx * ulb;
    return make_tuple(Re, At, Ca, Pe, dx, dt);
}

inline void printParameters_laplace2D_Liang(const Dim_laplace2D_Liang &dim, double Re, double At, double Ca, double Pe, double ulb, int N, double max_t) {
    cout << "Layered 2-D (Liang conservative Allen-Cahn - BGK/BGK)\n"
         << "N = " << N << '\n'
         << "nx = " << dim.nx << '\n'
         << "ny = " << dim.ny << '\n'
         << "Re = " << Re << '\n'
         << "At = " << At << '\n'
         << "Ca = " << Ca << '\n'
         << "Pe = " << Pe << '\n'
         << "ulb = " << ulb << '\n'
         << "max_t = " << max_t << '\n';
}

inline auto restartClock_laplace2D_Liang() {
    return make_pair(high_resolution_clock::now(), 0);
}

template <typename TP>
inline void printMlups_laplace2D_Liang(TP start, int iter, size_t nelem) {
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * static_cast<size_t>(iter)) / us.count();
    cout << "Runtime: " << us.count() / 1000000.0 << " seconds " << endl;
    cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

struct LBM_laplace2D_Liang {
    using CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem + 2 * 9 * nelem; }

    CellData* lattice;
    CellType_laplace2D_Liang* flag;
    int* parity;
    array<int,2>* c;
    int* opp;
    double* t;
    double* phiu_prev_x;
    double* phiu_prev_y;
    int* nb;
    int* nbb;
    double phi_l, phi_g, rho_l, rho_g, nu_l, nu_g, Sigma, Fbx, Fby, xi, M;
    Dim_laplace2D_Liang dim;

    static constexpr double cs2 = 1.0 / 3.0;
    static constexpr double PI = 3.1415926535897932384626433832795;

    inline size_t xyz_to_i(int x, int y) const { return static_cast<size_t>(y + dim.ny * x); }
    inline auto i_to_xyz(int i) const { return make_tuple(i / dim.ny, i % dim.ny); }

    inline double& fin (int i,int k) { return lattice[*parity * dim.npop + k * dim.nelem + i]; }
    inline double& fout(int i,int k) { return lattice[(1 - *parity) * dim.npop + k * dim.nelem + i]; }

    inline double& gin (int i,int k) { return lattice[*parity * dim.npop + 2 * dim.npop + k * dim.nelem + i]; }
    inline double& gout(int i,int k) { return lattice[(1 - *parity) * dim.npop + 2 * dim.npop + k * dim.nelem + i]; }

    inline double fin (int i,int k) const { return lattice[*parity * dim.npop + k * dim.nelem + i]; }
    inline double gin (int i,int k) const { return lattice[*parity * dim.npop + 2 * dim.npop + k * dim.nelem + i]; }

    // expects nb and nbb pointers to be valid (size nelem*9); periodic in x, non-periodic (wall) in y
    inline void build_neighbors() {
        for (int x = 0; x < dim.nx; ++x) {
            for (int y = 0; y < dim.ny; ++y) {
                int i = static_cast<int>(xyz_to_i(x, y));
                for (int k = 0; k < 9; ++k) {
                    int XX = (x + c[k][0] + dim.nx) % dim.nx;
                    int YY = y + c[k][1]; // non-periodic in y

                    int XXb = (x - c[k][0] + dim.nx) % dim.nx;
                    int YYb = y - c[k][1];

                    // clamp YY/YYb to stay in [0, ny-1]
                    if (YY  < 0)         YY  = 0;
                    if (YY  >= dim.ny)   YY  = dim.ny - 1;
                    if (YYb < 0)         YYb = 0;
                    if (YYb >= dim.ny)   YYb = dim.ny - 1;

                    nb [i * 9 + k] = static_cast<int>(xyz_to_i(XX,  YY ));
                    nbb[i * 9 + k] = static_cast<int>(xyz_to_i(XXb, YYb));
                }
            }
        }
    }

    // Initialization: layered liquid slab (phi hyperbolic-tangent profile), u = 0, P* = 0.
    // FIX: the interface half-width now uses the model's own diffuse-interface
    // parameter `xi` (read from the config file, used consistently everywhere else:
    // chemical_potential, lambda_phi, etc.) instead of a hardcoded local value.
    // Using a mismatched IC width causes an unnecessary transient as the profile
    // relaxes toward the equilibrium width dictated by kappa/beta (Eq. 7).
    inline void iniLattice(double& f0) {
        auto i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz(static_cast<int>(i));

        const double slab_frac = 0.4; // fraction of domain height occupied by the liquid slab

        const double H = double(dim.ny);
        const double Y = double(iY) + 0.5;

        const double yc = 0.5 * H;
        const double half_thickness = 0.5 * slab_frac * H;

        const double y_low  = yc - half_thickness;
        const double y_high = yc + half_thickness;

        // Smooth indicator: ~1 inside slab, ~0 outside. Half-width set by xi (Eq. 3/7).
        const double s_low  = 0.5 * (1.0 + std::tanh((Y - y_low ) / (0.5 * xi)));
        const double s_high = 0.5 * (1.0 - std::tanh((Y - y_high) / (0.5 * xi)));
        const double s_mid  = s_low * s_high;

        const double phi = s_mid * phi_l + (1.0 - s_mid) * phi_g;

        for (int k = 0; k < 9; ++k) {
            fin(i, k) = phi * t[k]; // f_k = phi * w_k
            gin(i, k) = 0.0;        // P* = 0
        }
    }

    inline double macro_phi(int i) const { // Eq. (12): phi = sum_i f_i
        double phi = 0.0;
        for (int k = 0; k < 9; ++k) phi += fin(i, k);
        return phi;
    }

    inline array<double,2> macro_u(int i) const {
        // Eq. (22a): rho u = sum_i c_i g_i + 0.5 dt (F_s + G)
        double phi = macro_phi(i);
        double rho = rho_from_phi(phi);
        array<double,2> mom{0.0, 0.0};
        for (int k = 0; k < 9; ++k) {
            mom[0] += c[k][0] * gin(i, k);
            mom[1] += c[k][1] * gin(i, k);
        }
        auto gradPhi = grad_phi(i);
        double mu_phi = chemical_potential(i);
        array<double,2> Fs{mu_phi * gradPhi[0], mu_phi * gradPhi[1]};
        mom[0] += 0.5 * (Fs[0] + Fbx);
        mom[1] += 0.5 * (Fs[1] + Fby);
        return {mom[0] / rho, mom[1] / rho};
    }

    // ============================================================
    // Liang 2018 material-property helpers
    // ============================================================

    // Eq. (13): rho = phi * (rho_l - rho_g) + rho_g
    inline double rho_from_phi(double phi) const {
        return rho_g + phi * (rho_l - rho_g);
    }

    // Eq. (24): nu = phi * (nu_l - nu_g) + nu_g
    inline double nu_from_phi(double phi) const {
        return nu_g + phi * (nu_l - nu_g);
    }

    // Eq. (23): nu = cs^2 (tau_g - 0.5) dt => tau_g = nu/(cs^2 dt) + 0.5
    inline double tau_g_from_phi(double phi) const {
        return nu_from_phi(phi) / cs2 + 0.5;
    }

    // Eq. (14): M = cs^2 (tau_f - 0.5) dt => tau_f = M/(cs^2 dt) + 0.5
    inline double tau_f_from_M() const {
        return M / cs2 + 0.5;
    }

    // Eq. (3): lambda = 4 phi (1-phi) / xi
    inline double lambda_phi(double phi) const {
        return 4.0 * phi * (1.0 - phi) / xi;
    }

    inline array<double,2> body_force() const { return {Fbx, Fby}; }

    // Eq. (2)
    inline array<double,2> unit_normal_from_gradphi(const array<double,2>& grad_phi) const {
        double s = sqrt(grad_phi[0] * grad_phi[0] + grad_phi[1] * grad_phi[1]);
        if (s < 1e-30) return {0.0, 0.0};
        return {grad_phi[0] / s, grad_phi[1] / s};
    }

    inline array<double,2> dphiu_dt(int i, double phi, const array<double,2>& u) const {
        return {(phi * u[0] - phiu_prev_x[i]), (phi * u[1] - phiu_prev_y[i])};
    }

    inline void store_phiu_prev(int i, double phi, const array<double,2>& u) {
        phiu_prev_x[i] = phi * u[0];
        phiu_prev_y[i] = phi * u[1];
    }

    // ============================================================
    // Liang 2018 differential operators
    // ============================================================

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
                int nbb_i = static_cast<int>(xyz_to_i(ixbb, iybb));
                phi_nb = macro_phi(nbb_i);
            } else {
                int nb_i = static_cast<int>(xyz_to_i(ix, iy));
                if (flag[nb_i] == CellType_laplace2D_Liang::bounce_back) {
                    int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                    int iybb = iY - c[k][1];
                    int nbb_i = static_cast<int>(xyz_to_i(ixbb, iybb));
                    phi_nb = macro_phi(nbb_i);
                } else {
                    phi_nb = macro_phi(nb_i);
                }
            }

            gphi[0] += t[k] * c[k][0] * phi_nb;
            gphi[1] += t[k] * c[k][1] * phi_nb;
        }

        gphi[0] /= cs2;
        gphi[1] /= cs2;
        return gphi;
    }

    // Eq. (28): lap(phi)(x) = sum_{i!=0} 2 w_i [phi(x+c_i dt)-phi(x)] / (cs^2 dt^2)
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
                int nbb_i = static_cast<int>(xyz_to_i(ixbb, iybb));
                phi_nb = macro_phi(nbb_i);
            } else {
                int nb_i = static_cast<int>(xyz_to_i(ix, iy));
                if (flag[nb_i] == CellType_laplace2D_Liang::bounce_back) {
                    int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                    int iybb = iY - c[k][1];
                    int nbb_i = static_cast<int>(xyz_to_i(ixbb, iybb));
                    phi_nb = macro_phi(nbb_i);
                } else {
                    phi_nb = macro_phi(nb_i);
                }
            }

            lap += 2.0 * t[k] * (phi_nb - phi_c);
        }

        lap /= cs2;
        return lap;
    }

    // Eq. (28) + Eq. (13)
    inline array<double,2> gradient_rho_from_gradphi(const array<double,2>& grad_phi) const {
        double delta_rho = rho_l - rho_g;
        return { delta_rho * grad_phi[0], delta_rho * grad_phi[1] };
    }

    // ============================================================
    // Liang 2018 interface geometry and surface tension
    // ============================================================

    // Eq. (2): n = grad(phi) / |grad(phi)|
    inline array<double,2> normal_phi(int i) const {
        auto gphi = grad_phi(i);
        double mag = std::sqrt(gphi[0] * gphi[0] + gphi[1] * gphi[1]);
        if (mag < 1e-14) return {0.0, 0.0};
        return {gphi[0] / mag, gphi[1] / mag};
    }

    // Eq. (6): mu_phi = 4 beta phi(phi-1)(phi-0.5) - kappa laplacian(phi)
    // Eq. (7): kappa = 3 sigma W / 2, beta = 12 sigma / W, with W = xi
    inline double chemical_potential(int i) const {
        double phi = macro_phi(i);
        double lap = laplacian_phi(i);
        double kappa = 1.5 * Sigma * xi;
        double beta  = 12.0 * Sigma / xi;
        return 4.0 * beta * phi * (phi - 1.0) * (phi - 0.5) - kappa * lap;
    }

    // Eq. (5): F_s = mu_phi * grad(phi)
    inline array<double,2> surface_force(int i) const {
        auto gphi = grad_phi(i);
        double mu_phi = chemical_potential(i);
        return {mu_phi * gphi[0], mu_phi * gphi[1]};
    }

    // Total interfacial + body force: F = F_s + G
    inline array<double,2> F_total_liang(double mu_phi_val, const array<double,2>& grad_phi) const {
        array<double,2> Fs{mu_phi_val * grad_phi[0], mu_phi_val * grad_phi[1]};
        auto Gb = body_force();
        return {Fs[0] + Gb[0], Fs[1] + Gb[1]};
    }

    // Eq. (21a)
    inline array<double,2> velocity_liang_from_macro(const array<double,2>& mom, const array<double,2>& Ftot, double rho) const {
        return { (mom[0] + 0.5 * Ftot[0]) / rho, (mom[1] + 0.5 * Ftot[1]) / rho };
    }

    // Eq. (21b) / Eq. (17): s0
    inline double pressure_liang(int i, double rho, const array<double,2>& u, const array<double,2>& grad_rho) const {
        double sum_nonzero = 0.0;
        for (int k = 0; k < 9; ++k) if (k != 4) sum_nonzero += gin(i, k);
        double s0 = t[4] * ( - (u[0] * u[0] + u[1] * u[1]) / (2.0 * cs2) );
        return cs2 / (1.0 - t[4]) * (sum_nonzero + 0.5 * (u[0] * grad_rho[0] + u[1] * grad_rho[1]) + rho * s0);
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
        double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
        double uu = u[0] * u[0] + u[1] * u[1];
        return t[k] * ( ck_u / cs2 + (ck_u * ck_u) / (2.0 * cs2 * cs2) - uu / (2.0 * cs2) );
    }

    // Eq. (18)
    // FIX 1: use the TOTAL force F = F_s + G (surface tension + body force),
    //        not just the constant body force (Fbx, Fby). Previously the
    //        surface-tension force was only added in macro_u() as a diagnostic
    //        correction and never actually entered the momentum collision,
    //        so the interfacial force never balanced the pressure jump.
    // FIX 2: removed the spurious extra `u_dot_grad_rho` term. The `tensor`
    //        term already contains the full u*grad(rho):(c c - cs2 I) expansion,
    //        so adding u.grad(rho) again double-counts and cancels the -cs2
    //        diagonal correction, breaking Eq. (18).
    inline double Gi_liang(int k, double tau_g,
                            const array<double,2>& u,
                            const array<double,2>& Ftot,
                            const array<double,2>& grad_rho) const {
        double ci_dot_F = c[k][0] * Ftot[0] + c[k][1] * Ftot[1];

        double tensor =
              (u[0] * grad_rho[0]) * (c[k][0] * c[k][0] - cs2)
            + (u[0] * grad_rho[1]) * (c[k][0] * c[k][1])
            + (u[1] * grad_rho[0]) * (c[k][1] * c[k][0])
            + (u[1] * grad_rho[1]) * (c[k][1] * c[k][1] - cs2);

        return (1.0 - 0.5 / tau_g) * t[k] * (ci_dot_F / cs2 + tensor / cs2);
    }

    inline void stream_tab(int i, int k, double pf, double pg) {
        const int j = nb[i * 9 + k];
        if (flag[j] == CellType_laplace2D_Liang::bounce_back) {
            fout(i, opp[k]) = pf;
            gout(i, opp[k]) = pg;
        } else {
            fout(j, k) = pf;
            gout(j, k) = pg;
        }
    }

    // Eq. (9) / Eq. (16)
    // FIX: Gi_liang is now called with Ftot (F_s + G) instead of the body
    //      force alone.
    inline tuple<double,double> collideBgkLiang(int i, int k, const array<double,2>& u, double rho, double phi,
                                                 double p, double tau_g, double tau_f,
                                                 const array<double,2>& Ftot,
                                                 const array<double,2>& phase_src,
                                                 const array<double,2>& grad_rho) const {
        double feq = t[k] * phi * (1.0 + (c[k][0] * u[0] + c[k][1] * u[1]) / cs2);
        double geq = (k == 4) ? (p / cs2) * (t[k] - 1.0) + rho * si_liang(k, u)
                               : (p / cs2) * t[k] + rho * si_liang(k, u);
        double Fi = Fi_liang(k, tau_f, phase_src);
        double Gi = Gi_liang(k, tau_g, u, Ftot, grad_rho); // FIX: Ftot passed through
        double pf = fin(i, k) - (fin(i, k) - feq) / tau_f + Fi;
        double pg = gin(i, k) - (gin(i, k) - geq) / tau_g + Gi;
        return {pf, pg};
    }

    inline void operator()(double& f0) {
        int i = static_cast<int>(&f0 - lattice);
        if (flag[i] != CellType_laplace2D_Liang::bulk) return;

        double phi = macro_phi(i);
        double rho = rho_from_phi(phi);
        double tau_g = tau_g_from_phi(phi);
        double tau_f_loc = tau_f_from_M();

        auto gradPhi = grad_phi(i);
        auto grad_rho = gradient_rho_from_gradphi(gradPhi);
        double mu_phi_val = chemical_potential(i);
        auto Ftot = F_total_liang(mu_phi_val, gradPhi); // FIX: Fs + G computed once, used in collision below

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

inline void saveVtkFields_laplace2D_Liang(LBM_laplace2D_Liang& lbm, int time_iter, double dx = 0.0) {
    Dim_laplace2D_Liang const& dim = lbm.dim;
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

    os << "SCALARS phi float 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = static_cast<int>(lbm.xyz_to_i(x, y));
            float val = 0.f;
            if (lbm.flag[i] == CellType_laplace2D_Liang::bulk) val = static_cast<float>(lbm.macro_phi(i));
            os << val << " ";
        }
        os << "\n";
    }

    os << "\nSCALARS density float 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = static_cast<int>(lbm.xyz_to_i(x, y));
            float val = 0.f;
            if (lbm.flag[i] == CellType_laplace2D_Liang::bulk) val = static_cast<float>(lbm.rho_from_phi(lbm.macro_phi(i)));
            os << val << " ";
        }
        os << "\n";
    }

    os << "\nSCALARS pressure float 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = static_cast<int>(lbm.xyz_to_i(x, y));
            double val = 0.0;
            if (lbm.flag[i] == CellType_laplace2D_Liang::bulk) {
                double phi = lbm.macro_phi(i), rho = lbm.rho_from_phi(phi);
                auto grad_phi = lbm.grad_phi(i);
                auto grad_rho = lbm.gradient_rho_from_gradphi(grad_phi);
                auto u = lbm.macro_u(i);
                val = lbm.pressure_liang(i, rho, u, grad_rho);
            }
            os << val << " ";
        }
        os << "\n";
    }

    os << "\nVECTORS velocity float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = static_cast<int>(lbm.xyz_to_i(x, y));
            double ux = 0.0, uy = 0.0;
            if (lbm.flag[i] == CellType_laplace2D_Liang::bulk) {
                auto u = lbm.macro_u(i);
                ux = u[0]; uy = u[1];
            }
            os << static_cast<float>(ux) << " " << static_cast<float>(uy) << " 0\n";
        }
    }

    os << "\nSCALARS Flag int 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x, y);
            os << ((lbm.flag[i] == CellType_laplace2D_Liang::bounce_back) ? 1 : 0) << " ";
        }
        os << "\n";
    }
}

inline double computeEnergy_laplace2D_Liang(LBM_laplace2D_Liang& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = static_cast<int>(lbm.xyz_to_i(x, y));
            if (lbm.flag[i] == CellType_laplace2D_Liang::bulk) {
                auto u = lbm.macro_u(i);
                energy += u[0] * u[0] + u[1] * u[1];
            }
        }
    }
    return 0.5 * energy;
}

inline void inigeom_laplace2D_Liang(LBM_laplace2D_Liang& lbm) {
    Dim_laplace2D_Liang const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = static_cast<int>(lbm.xyz_to_i(x, y));
            if (y == 0 || y == dim.ny - 1) {
                lbm.flag[i] = CellType_laplace2D_Liang::bounce_back;
                for (int k = 0; k < 9; ++k) { lbm.fin(i, k) = 0.0; lbm.gin(i, k) = 0.0; }
            } else {
                lbm.flag[i] = CellType_laplace2D_Liang::bulk;
            }
        }
    }
}

inline void laplace2D_Liang() {
    ifstream contfile("../apps/Config_Files/config_layered2D_Fakhari.txt");
    if (!contfile.is_open()) throw invalid_argument("Config file not found: config_layered2D_Fakhari.txt");

    double ulb = 1.0, max_t = 0.01, phi_l = 1.0, phi_g = 0.0, rho_l = 1.0, rho_g = 1e-3;
    double Sigma = 0.01, xi = 4.0, M = 0.02, Fbx = 0.0, Fby = 0.0;
    double mu_l = 0.1, mu_g = 0.0001;
    int N = 128, out_freq = 100, vtk_freq = 100;

    string line, param, value;
    while (getline(contfile, line)) {
        if (auto pos = line.find('#'); pos != string::npos) line.erase(pos);
        if (line.empty()) continue;
        istringstream ls(line);
        if (!(ls >> param >> value)) continue;
        if      (param == "N")        N        = stoi(value);
        else if (param == "max_t")    max_t    = stod(value);
        else if (param == "out_freq") out_freq = stoi(value);
        else if (param == "vtk_freq") vtk_freq = stoi(value);
        else if (param == "phi_l")    phi_l    = stod(value);
        else if (param == "phi_g")    phi_g    = stod(value);
        else if (param == "rho_l")    rho_l    = stod(value);
        else if (param == "rho_g")    rho_g    = stod(value);
        else if (param == "Sigma")    Sigma    = stod(value);
        else if (param == "xi")       xi       = stod(value);
        else if (param == "M")        M        = stod(value);
        else if (param == "ulb")      ulb      = stod(value);
        else if (param == "Fbx")      Fbx      = stod(value);
        else if (param == "Fby")      Fby      = stod(value);
        else if (param == "mu_l")     mu_l     = stod(value);
        else if (param == "mu_g")     mu_g     = stod(value);
    }

    Dim_laplace2D_Liang dim{10, N};
    auto [Re, At, Ca, Pe, dx, dt] = lbParameters_laplace2D_Liang(ulb, N, mu_l, Sigma, rho_l, rho_g, M);
    printParameters_laplace2D_Liang(dim, Re, At, Ca, Pe, ulb, N, max_t);

    vector<CellData> lattice_vect(LBM_laplace2D_Liang::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();
    vector<CellType_laplace2D_Liang> flag_vect(dim.nelem);
    CellType_laplace2D_Liang* flag = flag_vect.data();
    vector<int> parity_vect{0};
    int* parity = parity_vect.data();
    vector<int> nb_vect(dim.nelem * 9), nbb_vect(dim.nelem * 9);
    vector<double> phiu_prev_x_vect(dim.nelem, 0.0), phiu_prev_y_vect(dim.nelem, 0.0);

    auto [c_vect, opp_vect, t_vect] = d2q9_constants_laplace2D_Liang();
    LBM_laplace2D_Liang lbm{
        lattice, flag, parity, c_vect.data(), opp_vect.data(), t_vect.data(),
        phiu_prev_x_vect.data(), phiu_prev_y_vect.data(), nb_vect.data(), nbb_vect.data(),
        phi_l, phi_g, rho_l, rho_g, mu_l, mu_g, Sigma, Fbx, Fby, xi, M, dim
    };
    lbm.build_neighbors();
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });
    inigeom_laplace2D_Liang(lbm);

    auto [start, clock_iter] = restartClock_laplace2D_Liang();
    ofstream efile("energy.dat");
    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) saveVtkFields_laplace2D_Liang(lbm, time_iter);
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter << ", t = " << setprecision(6) << time_iter * dt << setprecision(3)
                 << " [" << (time_iter * dt / max_t * 100.) << "%]" << endl;
            double energy = computeEnergy_laplace2D_Liang(lbm) * dx * dx / (dt * dt);
            cout << "Average energy: " << setprecision(8) << energy << endl;
            efile << setw(12) << setprecision(6) << time_iter * dt << setw(18) << setprecision(8) << energy << "\n";
        }
        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        *parity = 1 - *parity;
        ++clock_iter;
    }
    printMlups_laplace2D_Liang(start, clock_iter, dim.nelem);
}
