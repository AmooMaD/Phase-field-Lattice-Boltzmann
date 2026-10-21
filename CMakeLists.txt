# ******************************************************************************
# CooLBM SOFTWARE LIBRARY
# (see license block in your original file)
# ******************************************************************************

cmake_minimum_required(VERSION 3.6)

# Name it however you like
project(COOLBM LANGUAGES CXX)

# -------- Build type (default Release) ----------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -------- Output dir: put binaries in ../out ----------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../out)

# -------- Language / warnings -------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Common warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wextra -Wpedantic)
  endif()
elseif (MSVC)
  add_compile_options(/W3)
endif()

message(STATUS "Compiling with ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER}). Build type: ${CMAKE_BUILD_TYPE}")

# -------- Performance flags (Release) -----------------------------------------
# GCC / Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math -funroll-loops -fno-math-errno -fno-trapping-math")
# Intel classic (icc/icpc) or oneAPI (icx/icpx)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  # xHost ~ march=native; icx also understands -ffast-math
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -xHost -ffast-math -funroll-loops")
# MSVC (rough equivalents)
elseif(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ot /GS- /GL /fp:fast /arch:AVX2")
endif()

# -------- LTO / IPO if supported ----------------------------------------------
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
if(ipo_ok)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  message(STATUS "IPO/LTO enabled for Release")
else()
  message(STATUS "IPO/LTO not available: ${ipo_msg}")
endif()

# -------- Optional CUDA (kept like your original) ------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  # Tune this to your GPU if you actually compile CUDA parts
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Use fast math in Release
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
  # Example arch; change if needed
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Ampere default; set as needed (e.g., 75 for Turing)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
endif()

# -------- oneTBB (required) ---------------------------------------------------
set(USE_TBB ON CACHE BOOL "Use oneTBB")
if(USE_TBB)
  find_package(TBB REQUIRED tbb)
  message(STATUS "Found oneTBB: ${TBB_DIR}")
endif()

# -------- Global compile-time knobs (tiling etc.) ------------------------------
# You can override from the command line:
#   -DCOOLBM_TILE_X=256 -DCOOLBM_TILE_Y=128
if(NOT DEFINED COOLBM_TILE_X)
  set(COOLBM_TILE_X 128)
endif()
if(NOT DEFINED COOLBM_TILE_Y)
  set(COOLBM_TILE_Y 128)
endif()
add_compile_definitions(COOLBM_TILE_X=${COOLBM_TILE_X} COOLBM_TILE_Y=${COOLBM_TILE_Y})

# Guard to turn on/off extra NaN checks (off by default)
# add_compile_definitions(COOLBM_NAN_GUARD=1)

# -------- Make TBB visible to subprojects -------------------------------------
# If your targets in apps/ link against TBB::tbb there is nothing else to do.
# If they DON'T, we still make the imported target visible here.
if(TBB_FOUND)
  # nothing else needed; targets can link PRIVATE or PUBLIC TBB::tbb
endif()

# -------- Add subdirectories (your actual code) --------------------------------
add_subdirectory(apps)

