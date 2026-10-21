#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:44:55 2026

@author: zadm8483
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Contact angle data
# ============================================================

target_angle = np.array([10, 30, 60, 90, 120, 150, 160])
measured_angle = np.array([12.14, 32.22, 60.28, 90.00, 121.89, 158.76, 174.41])

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

# Measured results
ax.plot(
    target_angle,
    measured_angle,
    'o-',
    linewidth=2,
    markersize=7,
    label='Measured'
)

# Ideal agreement line: measured = target
ax.plot(
    [0, 180],
    [0, 180],
    '--',
    linewidth=1.8,
    label='Measured = target'
)

# ============================================================
# Formatting
# ============================================================

ax.set_xlabel('Target contact angle (°)', fontsize=13)
ax.set_ylabel('Measured contact angle (°)', fontsize=13)

ax.set_xlim(0, 180)
ax.set_ylim(0, 180)

ax.set_aspect('equal', adjustable='box')

ax.set_xticks(np.arange(0, 181, 30))
ax.set_yticks(np.arange(0, 181, 30))

ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()

# Save figure
plt.savefig('contact_angle_validation.png', dpi=900, bbox_inches='tight')

plt.show()