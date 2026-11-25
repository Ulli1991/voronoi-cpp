#!/usr/bin/env python3
"""Linear growth rate of the smooth Kelvin-Helmholtz setup of McNally, Lyra & Passy (2012) as luna-cpp's hydro_kh2d runs it:
periodic unit box, rho(y) from 1 to 2 and u(y) from 0.5 to -0.5 through exponential ramps of width 0.025 at y = 1/4 and 3/4,
uniform pressure 2.5, gamma 5/3, perturbation ~ exp(i k x + s t) with k = 4 pi.
The linearised compressible Euler equations about (rho(y), u(y), p), Fourier collocation in y with ny points, dense eigenvalues:
the method of LUNA's luna/hydro/exact_kh_linear.py (uniform density there), with the continuity equation added.
Prints the largest Re(s) for several ny (it must not depend on ny) and, as a check, Michalke's 0.1897 for a nearly
incompressible tanh layer through the same matrix."""
import math
import sys

import numpy as np

GAMMA, LAYER, PRESSURE = 5.0 / 3.0, 0.025, 2.5


def profile(y, f1, f2):
    """hydro_kh2d.cpp: McNally eq. 1 for y in [0, 1), mirrored about 1/2; f1, f2 the outer and inner values."""
    h = np.where(y < 0.5, y, 1.0 - y)
    fm = 0.5 * (f1 - f2)
    return np.where(h < 0.25, f1 - fm * np.exp((h - 0.25) / LAYER), f2 + fm * np.exp((0.25 - h) / LAYER))


def growth(k, rho, u, p, gamma, Ly):
    """max Re(s) of d/dt (rho', u', v', p') = A (rho', u', v', p') on len(rho) Fourier points over [0, Ly)."""
    ny = len(rho)
    kap = 2.0 * math.pi * np.fft.fftfreq(ny, Ly / ny)
    kap[ny // 2] = 0.0
    D = np.real(np.fft.ifft(1j * kap[:, None] * np.fft.fft(np.eye(ny), axis=0), axis=0))
    I, Z, ik = np.eye(ny), np.zeros((ny, ny)), 1j * k
    U, R = np.diag(u), np.diag(rho)
    A = np.block([
        [-ik * U, -ik * R, -np.diag(D @ rho) - R @ D, Z],
        [Z, -ik * U, -np.diag(D @ u), -ik * np.diag(1.0 / rho)],
        [Z, Z, -ik * U, -np.diag(1.0 / rho) @ D],
        [Z, -gamma * p * ik * I, -gamma * p * D, -ik * U]])
    return float(np.max(np.linalg.eigvals(A).real))


if __name__ == "__main__":
    for ny in [int(a) for a in sys.argv[1:]] or [256, 512, 1024]:
        y = np.arange(ny) / ny
        s = growth(4.0 * math.pi, profile(y, 1.0, 2.0), profile(y, 0.5, -0.5), PRESSURE, GAMMA, 1.0)
        print(f"McNally setup, k = 4 pi, ny = {ny}: sigma = {s:.6f}")
    # check: Michalke (1964), U = (1 + tanh(y / L)) / 2: fastest mode at k L = 0.4446 with s L / dU = 0.1897. Our layers are
    # tanh((y - y0) / d) between -dU/2 and +dU/2, so L = 2 d: s * 2 d / dU must give 0.1897 (two far layers, Mach 0.0025).
    ny, Ly, d, dU = 512, 2.0, 0.4446 / (2 * math.pi), 1.0
    y = np.arange(ny) * Ly / ny
    u = 0.5 * dU * (np.tanh((y - 0.5) / d) - np.tanh((y - 1.5) / d) - 1.0)
    s = growth(2.0 * math.pi, np.ones(ny), u, 1.0 * 200.0 ** 2 / GAMMA, GAMMA, Ly)
    print(f"check, two far tanh layers at Mach 0.0025: sigma 2 d / dU = {s * 2 * d / dU:.4f} (Michalke 1964: 0.1897)")
