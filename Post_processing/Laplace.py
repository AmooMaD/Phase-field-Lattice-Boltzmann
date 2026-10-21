#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 09:16:56 2026

@author: zadm8483
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Laplace law validation
# ============================================================

# Droplet radius
R = np.array([15, 20, 25, 30], dtype=float)

# ============================================================
# Numerical pressure difference
# ============================================================

deltaP_Fakhari = np.array([
    #0.000985,
    0.0006502,
    0.000486,
    0.000390,
    0.000325
])

deltaP_Liang = np.array([
    #0.000975,
    0.000640,
    0.000480,
    0.000382,
    0.000319
])

deltaP_HCZ = np.array([
    0.00651076 - 0.00651038,
    0.00650927 - 0.00650899,
    0.00650741 - 0.00650719,
    0.00650518 - 0.00650499
    
])

# ============================================================
# Surface tension
# ============================================================

sigma_Fakhari = 0.01
sigma_Liang   = 0.01
sigma_HCZ     = 5e-4 * 0.0111

# ============================================================
# Normalization
# ============================================================

laplace_Fakhari = deltaP_Fakhari / sigma_Fakhari
laplace_Liang   = deltaP_Liang   / sigma_Liang
laplace_HCZ     = deltaP_HCZ     / sigma_HCZ

# ============================================================
# Analytical Laplace law
# ============================================================

R_theory = np.linspace(100, 12, 10)
laplace_theory = 1.0 / R_theory

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7,6))

# Analytical
plt.plot(
    1.0/R_theory,
    laplace_theory,
    'k-',
    lw=2.5,
    label='Analytical'
)

# Fakhari
plt.plot(
    1.0/R,
    laplace_Fakhari,
    'ro',
    ms=6,
    label='Modeled (Fakhari et al. (2017))'
)

# Liang
plt.plot(
    1.0/R,
    laplace_Liang,
    'bo',
    ms=6,
    label='Modeled (Liang et al. (2018))'
)

# HCZ
plt.plot(
    1.0/R,
    laplace_HCZ,
    'go',
    ms=6,
    label='Modeled (He et al. (1999))'
)

plt.xlabel(r'$1/R$ (1/lu)', fontsize=14)
plt.ylabel(r'$\Delta P/\sigma$ (1/lu)', fontsize=14)

plt.legend()
plt.tight_layout()

plt.savefig('laplace.png', dpi=900, bbox_inches='tight')
plt.show()