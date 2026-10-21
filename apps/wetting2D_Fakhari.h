#pragma once
/*
  Complete D2Q9 conservative-phase-field LBM sessile-drop/contact-angle test
  WITH contact-angle measurement integrated.

  Model basis:
    A. Fakhari and D. Bolster, JCP 334 (2017), 620-638.
  This is the flat-wall test: periodic x, stationary walls at y=0 and y=ny-1.
  The target static contact angle theta_deg is measured through phi_l (liquid).

  Required config file path:
    ../apps/Config_Files/config_wetting2D_Fakhari.txt

  Entry point:
    wetting2D_Fakhari_with_measurement();
*/

#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <cstdint>

using namespace std;

namespace wetting2d_fakhari_meas {

static constexpr int Q = 9;
static constexpr double cs2 = 1.0 / 3.0;
static constexpr double pi = 3.1415926535897932384626433832795;

enum class CellType : uint8_t { solid, fluid };

struct Params {
    int nx = 128;
    int ny = 64;
    int max_steps = 200000;
    int out_freq = 1000;
    int vtk_freq = 5000;
    int measure_freq = 1000;

    double phi_l = 1.0;
    double phi_g = 0.0;
    double rho_l = 1.0;
    double rho_g = 1.0e-3;
    double mu_l = 0.1;
    double mu_g = 1.0e-3;
    double sigma = 0.01;
    double xi = 4.0;
    double mobility = 0.02;
    double theta_deg = 60.0;

    double R0 = 16.0;
    double X0 = -1.0;
    double Y0 = -1.0;
};

struct Lattice {
    int nx, ny, n;
    array<int, Q> cx{{-1, 0,-1,-1, 0, 1, 0, 1, 1}};
    array<int, Q> cy{{ 0,-1,-1, 1, 0, 0, 1, 1,-1}};
    array<int, Q> opp{{ 5, 6, 7, 8, 4, 0, 1, 2, 3}};
    array<double, Q> w{{1./9.,1./9.,1./36.,1./36.,4./9.,1./9.,1./9.,1./36.,1./36.}};

    vector<CellType> flag;
    vector<double> h[2];
    vector<double> g[2];
    int parity = 0;

    Params p;

    explicit Lattice(const Params& prm)
        : nx(prm.nx), ny(prm.ny), n(prm.nx * prm.ny), flag(n, CellType::fluid), p(prm) {
        h[0].assign(Q * n, 0.0); h[1].assign(Q * n, 0.0);
        g[0].assign(Q * n, 0.0); g[1].assign(Q * n, 0.0);
    }

    int id(int x, int y) const { return y + ny * x; }
    int wrapx(int x) const { x %= nx; return x < 0 ? x + nx : x; }
    int qid(int q, int i) const { return q * n + i; }

    double& hin(int q, int i)  { return h[parity][qid(q, i)]; }
    double& hout(int q, int i) { return h[1 - parity][qid(q, i)]; }
    double& gin(int q, int i)  { return g[parity][qid(q, i)]; }
    double& gout(int q, int i) { return g[1 - parity][qid(q, i)]; }
    double hin(int q, int i) const  { return h[parity][qid(q, i)]; }
    double gin(int q, int i) const  { return g[parity][qid(q, i)]; }

    double phi(int i) const {
        double s = 0.0;
        for (int q = 0; q < Q; ++q) s += hin(q, i);
        return s;
    }

    double pstar(int i) const {
        double s = 0.0;
        for (int q = 0; q < Q; ++q) s += gin(q, i);
        return s;
    }

    double rho_from_phi(double ph) const {
        const double a = (ph - p.phi_g) / (p.phi_l - p.phi_g);
        return p.rho_g + a * (p.rho_l - p.rho_g);
    }

    double mu_from_phi(double ph) const {
        const double a = (ph - p.phi_g) / (p.phi_l - p.phi_g);
        return p.mu_g + a * (p.mu_l - p.mu_g);
    }

    double tau_from_phi(double ph) const {
        const double rho = rho_from_phi(ph);
        const double mu = mu_from_phi(ph);
        return mu / (rho * cs2);
    }

    array<double,2> raw_u(int i) const {
        double ux = 0.0, uy = 0.0;
        for (int q = 0; q < Q; ++q) {
            ux += gin(q, i) * cx[q];
            uy += gin(q, i) * cy[q];
        }
        return {ux, uy};
    }

    // Bottom wall is at y=0. The phase-field ghost is constructed from the
    // prescribed normal derivative: dphi/dy = |grad(phi)| cos(theta).
    // theta is measured through liquid (phi_l). At theta=90 deg this is a
    // homogeneous Neumann condition.
    double bottom_ghost_phi(int x) const {
        const int i1 = id(x, 1);
        const int il = id(wrapx(x - 1), 1);
        const int ir = id(wrapx(x + 1), 1);
        const double ph1 = phi(i1);
        const double dphidx = 0.5 * (phi(ir) - phi(il));
        const double theta = p.theta_deg * pi / 180.0;
        const double sn = max(abs(sin(theta)), 1.0e-8);
        const double gradmag = abs(dphidx) / sn;
        const double dphidy = gradmag * cos(theta);
        // wall is half a lattice step below y=1: ph(0) = ph(1) - dphidy
        const double ghost = ph1 - dphidy;
        const double dphi = p.phi_l - p.phi_g;
        return max(p.phi_g - 0.25*dphi, min(p.phi_l + 0.25*dphi, ghost));
    }

    double top_ghost_phi(int x) const {
        // Neutral top boundary.
        return phi(id(x, ny - 2));
    }

    // Phase value used by the derivative stencil. x is periodic; y is not.
    double stencil_phi(int x, int y) const {
        x = wrapx(x);
        if (y == 0) return bottom_ghost_phi(x);
        if (y == ny - 1) return top_ghost_phi(x);
        return phi(id(x, y));
    }

    array<double,2> grad_phi(int x, int y) const {
        double gx = 0.0, gy = 0.0;
        for (int q = 0; q < Q; ++q) {
            const double ph = stencil_phi(x + cx[q], y + cy[q]);
            gx += w[q] * cx[q] * ph;
            gy += w[q] * cy[q] * ph;
        }
        return {gx / cs2, gy / cs2};
    }

    double lap_phi(int x, int y) const {
        const double phc = phi(id(x, y));
        double s = 0.0;
        for (int q = 0; q < Q; ++q)
            s += w[q] * (stencil_phi(x + cx[q], y + cy[q]) - phc);
        return 2.0 * s / cs2;
    }

    double chemical_potential(int x, int y) const {
        const double ph = phi(id(x, y));
        const double beta = 12.0 * p.sigma / p.xi;
        const double kappa = 1.5 * p.sigma * p.xi;
        const double ph0 = 0.5 * (p.phi_l + p.phi_g);
        return 4.0 * beta * (ph - p.phi_g) * (ph - p.phi_l) * (ph - ph0)
             - kappa * lap_phi(x, y);
    }

    array<double,2> phase_force_direction(int x, int y) const {
        const int i = id(x, y);
        const double ph = phi(i);
        const auto grad = grad_phi(x, y);
        const double norm = sqrt(grad[0]*grad[0] + grad[1]*grad[1]) + 1.0e-30;
        const double ph0 = 0.5 * (p.phi_l + p.phi_g);
        const double fac = (1.0 - 4.0*(ph - ph0)*(ph - ph0)
                            / ((p.phi_l - p.phi_g)*(p.phi_l - p.phi_g))) / p.xi;
        return {fac * grad[0] / norm, fac * grad[1] / norm};
    }

    array<double,2> total_force(int x, int y) const {
        const int i = id(x, y);
        const double ph = phi(i);
        const double rho = rho_from_phi(ph);
        const auto grad = grad_phi(x, y);
        const double muchem = chemical_potential(x, y);
        const double drho_dphi = (p.rho_l - p.rho_g) / (p.phi_l - p.phi_g);
        const double ps = pstar(i);

        // F_s + F_p. No body force in the static sessile-drop benchmark.
        return {muchem * grad[0] - ps * cs2 * drho_dphi * grad[0],
                muchem * grad[1] - ps * cs2 * drho_dphi * grad[1]};
    }

    array<double,2> velocity(int x, int y) const {
        const int i = id(x, y);
        const double rho = rho_from_phi(phi(i));
        const auto u0 = raw_u(i);
        const auto F = total_force(x, y);
        return {u0[0] + 0.5 * F[0] / rho,
                u0[1] + 0.5 * F[1] / rho};
    }

    void geometry() {
        for (int x = 0; x < nx; ++x) {
            flag[id(x, 0)] = CellType::solid;
            flag[id(x, ny - 1)] = CellType::solid;
        }
    }

    void initialize() {
        geometry();
        const double x0 = p.X0 >= 0.0 ? p.X0 : 0.5 * nx;
        const double y0 = p.Y0 >= 0.0 ? p.Y0 : 1.0 + p.R0;
        const double ph0 = 0.5 * (p.phi_l + p.phi_g);
        const double amp = 0.5 * (p.phi_l - p.phi_g);

        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                const int i = id(x, y);
                if (flag[i] == CellType::solid) continue;
                double dx = (x + 0.5) - x0;
                if (dx >  0.5*nx) dx -= nx;
                if (dx < -0.5*nx) dx += nx;
                const double dy = (y + 0.5) - y0;
                const double r = sqrt(dx*dx + dy*dy);
                const double ph = ph0 + amp * tanh((p.R0 - r) / (0.5 * p.xi));
                for (int q = 0; q < Q; ++q) {
                    hin(q, i) = w[q] * ph;
                    gin(q, i) = 0.0;
                }
            }
        }
    }

    void collide_and_stream() {
        fill(h[1 - parity].begin(), h[1 - parity].end(), 0.0);
        fill(g[1 - parity].begin(), g[1 - parity].end(), 0.0);

        const double tau_h = p.mobility / cs2;
        const double invh = 1.0 / (tau_h + 0.5);

        for (int x = 0; x < nx; ++x) {
            for (int y = 1; y < ny - 1; ++y) {
                const int i = id(x, y);
                const double ph = phi(i);
                const double ps = pstar(i);
                const double rho = rho_from_phi(ph);
                const double tau_g = tau_from_phi(ph);
                const double invg = 1.0 / (tau_g + 0.5);
                const auto u = velocity(x, y);
                const auto F = total_force(x, y);
                const auto Fph = phase_force_direction(x, y);
                const double usq2cs2 = (u[0]*u[0] + u[1]*u[1]) / (2.0 * cs2);

                for (int q = 0; q < Q; ++q) {
                    const double eu = cx[q]*u[0] + cy[q]*u[1];
                    const double gamma = w[q] * (1.0 + eu/cs2
                                       + eu*eu/(2.0*cs2*cs2) - usq2cs2);
                    const double heq = ph * gamma;
                    const double geq = ps*w[q] + gamma - w[q];
                    const double Sh = w[q] * (Fph[0]*cx[q] + Fph[1]*cy[q]);
                    const double Sg = w[q] * (F[0]*cx[q] + F[1]*cy[q]) / (rho*cs2);

                    const double hpost = hin(q, i) - invh*(hin(q, i) - (heq - 0.5*Sh)) + Sh;
                    const double gpost = gin(q, i) - invg*(gin(q, i) - (geq - 0.5*Sg)) + Sg;

                    const int xn = wrapx(x + cx[q]);
                    const int yn = y + cy[q];
                    if (yn <= 0 || yn >= ny - 1) {
                        // Hydrodynamic and phase-field distributions: halfway bounce-back.
                        hout(opp[q], i) = hpost;
                        gout(opp[q], i) = gpost;
                    } else {
                        const int j = id(xn, yn);
                        hout(q, j) = hpost;
                        gout(q, j) = gpost;
                    }
                }
            }
        }
        parity = 1 - parity;
    }

    double liquid_mass() const {
        double m = 0.0;
        for (int x = 0; x < nx; ++x)
            for (int y = 1; y < ny - 1; ++y)
                m += (phi(id(x,y)) - p.phi_g) / (p.phi_l - p.phi_g);
        return m;
    }

    void write_vtk(int step) const {
        ostringstream name;
        name << "wetting_" << setw(7) << setfill('0') << step << ".vtk";
        ofstream out(name.str());
        if (!out) throw runtime_error("Cannot write VTK file.");

        out << "# vtk DataFile Version 2.0\n";
        out << "Fakhari conservative phase-field wetting test\nASCII\n";
        out << "DATASET STRUCTURED_POINTS\n";
        out << "DIMENSIONS " << nx << ' ' << ny << " 1\n";
        out << "ORIGIN 0 0 0\nSPACING 1 1 1\n";
        out << "POINT_DATA " << n << "\n";

        out << "SCALARS phi float 1\nLOOKUP_TABLE default\n";
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int i = id(x,y);
                out << (flag[i] == CellType::solid ? p.phi_g : phi(i)) << '\n';
            }
        }

        out << "SCALARS density float 1\nLOOKUP_TABLE default\n";
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int i = id(x,y);
                out << (flag[i] == CellType::solid ? 0.0 : rho_from_phi(phi(i))) << '\n';
            }
        }

        out << "VECTORS velocity float\n";
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int i = id(x,y);
                if (flag[i] == CellType::solid) out << "0 0 0\n";
                else {
                    const auto u = velocity(x,y);
                    out << u[0] << ' ' << u[1] << " 0\n";
                }
            }
        }
    }

    // Contact-angle measurement (geometric reconstruction)
    void measureContactAngle(int step) const {
        const int iY1 = 1;  // first fluid row
        const int iX1 = nx / 2;
        int iX2 = 0;
        int iY2 = 0;
        const double rho_cut = 0.5 * (p.rho_l + p.rho_g);

        for (int iX = 0; iX < nx; ++iX) {
            const int i = id(iX, iY1);
            const double ph = phi(i);
            const double rho = rho_from_phi(ph);
            if (rho > rho_cut) iX2 += 1;
        }

        for (int iY = iY1; iY < ny - 1; ++iY) {
            const int i = id(iX1, iY);
            const double ph = phi(i);
            const double rho = rho_from_phi(ph);
            if (rho > rho_cut) iY2 += 1;
        }

        double R = 0.0;
        double teta = 0.0;
        if (iY2 > 0 && iX2 > 0) {
            R = (4.0 * iY2 * iY2 + iX2 * iX2) / (8.0 * iY2);
            const double denom = R - iY2;
            if (std::abs(denom) > 1.0e-12) {
                teta = std::atan( static_cast<double>(iX2) / (2.0 * denom) );
            } else {
                teta = 0.5 * pi;
            }
            teta *= 180.0 / pi;
            if (teta < 0.0) teta += 180.0;
        }

        cout.precision(8);
        cout << "[step " << step << "] "
             << "Base = " << iX2
             << ", Height = " << iY2
             << ", Contact Angle = " << teta << " deg\n";
    }
};

inline Params read_config(const string& filename) {
    ifstream in(filename);
    if (!in) throw invalid_argument("Cannot open config file: " + filename);
    Params p;
    string line, key, value;
    while (getline(in, line)) {
        const auto pos = line.find('#');
        if (pos != string::npos) line.erase(pos);
        istringstream is(line);
        if (!(is >> key >> value)) continue;
        if      (key == "nx")          p.nx = stoi(value);
        else if (key == "ny")          p.ny = stoi(value);
        else if (key == "max_steps")   p.max_steps = stoi(value);
        else if (key == "out_freq")    p.out_freq = stoi(value);
        else if (key == "vtk_freq")    p.vtk_freq = stoi(value);
        else if (key == "measure_freq") p.measure_freq = stoi(value);
        else if (key == "phi_l")       p.phi_l = stod(value);
        else if (key == "phi_g")       p.phi_g = stod(value);
        else if (key == "rho_l")       p.rho_l = stod(value);
        else if (key == "rho_g")       p.rho_g = stod(value);
        else if (key == "mu_l")        p.mu_l = stod(value);
        else if (key == "mu_g")        p.mu_g = stod(value);
        else if (key == "sigma")       p.sigma = stod(value);
        else if (key == "xi")          p.xi = stod(value);
        else if (key == "mobility")    p.mobility = stod(value);
        else if (key == "theta_deg")   p.theta_deg = stod(value);
        else if (key == "R0")          p.R0 = stod(value);
        else if (key == "X0")          p.X0 = stod(value);
        else if (key == "Y0")          p.Y0 = stod(value);
        else cerr << "Warning: unknown config key: " << key << '\n';
    }
    if (p.nx < 8 || p.ny < 8) throw invalid_argument("nx and ny must be >= 8.");
    if (p.xi <= 0.0 || p.mobility <= 0.0 || p.sigma <= 0.0)
        throw invalid_argument("xi, mobility, and sigma must be positive.");
    return p;
}

} // namespace wetting2d_fakhari_meas

inline void wetting2D_Fakhari() {
    using namespace wetting2d_fakhari_meas;
    const string config = "../apps/Config_Files/config_wetting2D_Fakhari.txt";
    Params p = read_config(config);
    Lattice lbm(p);
    lbm.initialize();

    cout << "Fakhari conservative phase-field wetting test (with measurement)\n"
         << "domain = " << p.nx << " x " << p.ny << '\n'
         << "theta_deg (target) = " << p.theta_deg << '\n'
         << "xi = " << p.xi << ", M = " << p.mobility
         << ", sigma = " << p.sigma << '\n'
         << "rho_l/rho_g = " << p.rho_l / p.rho_g
         << ", mu_l/mu_g = " << p.mu_l / p.mu_g << '\n';

    const double m0 = lbm.liquid_mass();
    const auto start = chrono::high_resolution_clock::now();
    for (int step = 0; step <= p.max_steps; ++step) {
        if (p.vtk_freq > 0 && step % p.vtk_freq == 0) lbm.write_vtk(step);
        if (p.out_freq > 0 && step % p.out_freq == 0) {
            const double m = lbm.liquid_mass();
            cout << "step " << setw(8) << step
                 << "  mass=" << setprecision(12) << m
                 << "  rel_mass_change=" << (m - m0) / m0 << '\n';
        }
        if (p.measure_freq > 0 && step % p.measure_freq == 0) {
            lbm.measureContactAngle(step);
        }
        if (step == p.max_steps) break;
        lbm.collide_and_stream();
    }
    const auto end = chrono::high_resolution_clock::now();
    const double sec = chrono::duration<double>(end - start).count();
    const double mlups = double(p.nx) * p.ny * p.max_steps / max(sec, 1.0e-30) / 1.0e6;
    cout << "Runtime: " << sec << " s, " << mlups << " MLUPS\n";
}