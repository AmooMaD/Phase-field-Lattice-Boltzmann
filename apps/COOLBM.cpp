/*******************************************************************************
# CooLBM SOFTWARE LIBRARY

# Copyright ©️ 2025 CORIA Lab., CNRS UMR 6614
# Contact: Dr. Mostafa Safdari Shadloo, msshadloo@coria.fr

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# In case of use of CooLBM, please properly cite the following articles:
# [1] R. Alamian, A. K. Nayak, M. S. Shadloo, CooLBM: A Collaborative Open-Source Reactive Multi-Phase/Component Simulation Code via Lattice Boltzmann Method, doi.org/10.48550 arXiv.2502.12955.
# [2] R. Alamian, M. Sawaf, C. Stockinger, A. Hadjadj, J. Latt, M. S. Shadloo, Modeling soot filter regeneration process through surface-reactive flow in porous media using iterative lattice Boltzmann method, Energy, 2024 (289), 129980.
# [3] C. Stockinger, A. Raiolo, R. Alamian, A. Hadjadj, U. Nieken, M. S. Shadloo, Lattice Boltzmann simulations of heterogeneous combustion reactions for application in porous media, Engineering Analysis with Boundary Elements, 2024 (166) 105817.
# [4] A. K. Nayak, A. Singh, M. Mesgarpour, M. S. Shadloo, A Numerical Investigation of Particle Deposition on a Substrate, doi.org/10.48550/arXiv.2502.12719.
# ******************************************************************************/

/// The double-population scheme allocates a second array of size Ntot × 19, hereafter called out. While this strategy doubles the total 
/// memory requirements, it simplifies the algorithm, as both the fk(t) and fout(t) values are stored at the memory location popk, while the k
/// streamed variables fk(t + 1) are kept separately in outk. Therefore, the collision and streaming steps can be fused 
/// (i.e. they are executed within a single memory traversal), as any access conflicts are naturally avoided. After a time 
/// iteration, an exchange of the arrays pop and out guarantees that the streamed, temporary populations are reused for the subsequent cycle.

// Vector is a sequential container to store elements and not index based. 
// Vector is dynamic in nature so, size increases with insertion of elements.
#include <vector>
// Array stores a fixed-size sequential collection of elements of the same type and 
// it is index based. As array is fixed size, once initialized can’t be resized.
#include <array>
#include <string>
// Header that defines the standard input/output stream objects (cin, cout, cerr, clog).
#include <iostream>
// Input/output stream class to operate on files.
#include <fstream>
// Is a library that is used to manipulate the output of C++ program. Using C++, header providing parametric manipulators
#include <iomanip>
// Defines a collection of functions especially designed to be used on ranges of elements (for_each)
#include <algorithm>
// This header is part of the algorithm library (seq, par, par_unseq, unseq)
#include <execution>
// The chrono library, a flexible collection of types that track time with varying degrees of precision (clocks, time points, durations)
#include <chrono>
// Header <cmath> declares a set of functions to compute common mathematical operations and transformations.
#include <cmath>
#include <tuple>
#include <stdexcept>



//#include "laplace2D_Fakhari_optimized.h"
//#include "laplace2D_liang.h"
//#include "laplace2D_Fakhari.h"


//#include "layered2D_Fakhari.h"
#include "layered2D_liang.h"

//#include "RTI_Liang.h"


//#include "wetting2D_Fakhari.h"


//#include "rayleighTaylor2D_Zhao.h"
//#include "porous2D_Zhao.h"
//#include "rayleighTaylor2D_Fakhari.h"
//#include "porousMedia2D_Fakhari.h"
//#include "laplace2D_Fakhari.h"
//#include "porous3D_Fakhari.h"
//#include "multiphase_porous2D.h"
//#include "porous2D.h"
//#include "free_injection_2D.h"

//#include "laplace2d_mrt.h"
//#include "laplace2D_MRT_opt.h"
//#include "layered2D_MRT.h"
//#include "rayleighTaylor2D_MRT.h"
//#include "laplace2D_Fakhari_optimized.h"
//#include "RTI2D_optimized_standard.h"
//#include "RTI2D_Fakhari_optimized.h"
//#include "RTI2D_MRT_optimized.h"
using namespace std;
using namespace std::chrono;

// Note: certain functions are similar in "combustion2D"
// with "Flow_2D_SRT_ParticleDeposition" and "Particle_2D_SRT_ParticleDeposition"
// while doing simulation for "combustion2D", do comment to
// #include "Flow_2D_SRT_ParticleDeposition.h" and #include "Particle_2D_SRT_ParticleDeposition.h"
// and also "runFlowfield_2D_SRT_ParticleDeposition()" and "runParticlefield_2D_SRT_ParticleDeposition()"
// if compilation error arises after this, you can move
// "Flow_2D_SRT_ParticleDeposition", "Deposition_2D_SRT_Utility", and "Particle_2D_SRT_ParticleDeposition"
// in "apps" folder to a different location
// conversly same thing is applied for performing "Particle Deposition" simulation
// or another solution is
// don't "Flow_2D_SRT_ParticleDeposition", "Deposition_2D_SRT_Utility", and "Particle_2D_SRT_ParticleDeposition"
// and "lbm_combustion2D" and "utility_combustion2D" in the same folder for any simulations


// Here the problem that will be solved should be selected from the following list:
// "combustion2D", "couette3D", "poiseuille3D", "rayleighTaylor2D", "taylorGreen3D",
// "mixing2D", "contactAngle2D", "cavitation2D", "Flow_2D_SRT_ParticleDeposition"
// "Particle_2D_SRT_ParticleDeposition" 
string problem = "laplace2D_Liang";

int main() {

    //if (problem == "laplace2D_Fakhari") {
    //    laplace2D_Fakhari();
    //}

    //if (problem == "laplace2D_MRT_opt") {
    //    laplace2D_MRT_opt();
   // }

    //if (problem == "layered2D_MRT") {
    //    layered2D_MRT();
    //}

    if (problem == "laplace2D_Liang") {
        laplace2D_Liang();
    }

   //if (problem == "rayleighTaylor2D_MRT") {
   //    rayleighTaylor2D_MRT();
   //}
    
    //if (problem == "laplace2D_Liang") {
    //    laplace2D_Liang();
    //}

    //if (problem == "rti2d_MRT_optimized") {
    //    rti2d_MRT_optimized();
    //}

    //if (problem == "rayleighTaylor2D_Fakhari") {
    //    rayleighTaylor2D_Fakhari();
    //}

    //if (problem == "layered2D_Fakhari") {
    //    layered2D_Fakhari();
    //}
    return 0;
}
