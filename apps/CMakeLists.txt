# ******************************************************************************
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
# ******************************************************************************

cmake_minimum_required(VERSION 3.6)

#add_executable(combustion2D_vop combustion2D_vop.cpp)
add_executable(COOLBM COOLBM.cpp)
#add_executable(poissuelle3D poissuelle3D.cpp)
#add_executable(couette3D couette3D.cpp)
#add_executable(contactAngle contactAngle.cpp)
#add_executable(contactAngle2 contactAngle2.cpp)
#add_executable(contactAngle3 contactAngle3.cpp)
#add_executable(MRT6_final MRT6_final.cpp)

if (USE_TBB)
   #target_link_libraries(combustion2D_vop "tbb")
   #target_link_libraries(COOLBM "tbb")
   target_link_libraries(COOLBM PRIVATE TBB::tbb)
   #target_link_libraries(poissuelle3D "tbb")
   #target_link_libraries(couette3D "tbb")
   #target_link_libraries(contactAngle "tbb")
   #target_link_libraries(contactAngle2 "tbb")
   #target_link_libraries(contactAngle3 "tbb")
   #target_link_libraries(MRT6_final "tbb")
   
endif()

