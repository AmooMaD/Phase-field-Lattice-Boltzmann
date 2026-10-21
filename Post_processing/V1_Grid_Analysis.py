#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:56:36 2026

@author: zadm8483
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Mesh resolution
# ============================================================

N = np.array([100, 200, 400, 800])

# Relative L2 error (%)
error_Fakhari = np.array([
    3.948787,
    2.145550,
    1.116147,
    0.5686782
])

error_Liang = np.array([
    3.948787,
    2.145532,
    1.115842,
    0.5687623
])

error_HCZ = np.array([
    3.948787,
    2.145576,
    1.116119,
    0.5688390
])


# ============================================================
# Plot mesh independence
# ============================================================

plt.figure(figsize=(7, 6))

plt.plot(
    N,
    error_Fakhari,
    'ro-',
    ms=6,
    lw=1.8,
    label='Model (Fakhari et al. 2017)'
)

plt.plot(
    N,
    error_Liang,
    'bo-',
    ms=6,
    lw=1.8,
    label='Model (Liang et al. 2018)'
)

plt.plot(
    N,
    error_HCZ,
    'go-',
    ms=6,
    lw=1.8,
    label='Model (He et al. 1999)'
)

plt.xlabel('Mesh resolution, N', fontsize=14)
plt.ylabel('Relative L2 error (%)', fontsize=14)

plt.xticks(N)

plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 6))

plt.loglog(
    N,
    error_Fakhari,
    'ro-',
    ms=6,
    lw=1.8,
    label='Model (Fakhari et al. 2017)'
)

plt.loglog(
    N,
    error_Liang,
    'bo-',
    ms=6,
    lw=1.8,
    label='Model (Liang et al. 2018)'
)

plt.loglog(
    N,
    error_HCZ,
    'go-',
    ms=6,
    lw=1.8,
    label='Model (He et al. 1999)'
)

plt.xlabel('Mesh resolution, N', fontsize=14)
plt.ylabel('Relative L2 error (%)', fontsize=14)

plt.legend()
plt.tight_layout()


plt.savefig('D1_V1_grid_dependency.png', dpi=900, bbox_inches='tight')
plt.show()