#!/usr/bin/env python3
"""Refine a Misiurewicz point c (z_{k+p} == z_k, z_0 = 0) to high precision.

Used once to produce the default zoom centre shared by all implementations.
Usage: misiurewicz.py PREPERIOD PERIOD RE IM [DIGITS]
"""
import sys
import mpmath as mp


def orbit(c, n):
    z, dz = mp.mpc(0), mp.mpc(0)
    zs, dzs = [z], [dz]
    for _ in range(n):
        z, dz = z * z + c, 2 * z * dz + 1
        zs.append(z)
        dzs.append(dz)
    return zs, dzs


def main():
    k, p = int(sys.argv[1]), int(sys.argv[2])
    c = mp.mpc(sys.argv[3], sys.argv[4])
    digits = int(sys.argv[5]) if len(sys.argv) > 5 else 340
    mp.mp.dps = digits + 40
    for _ in range(200):
        zs, dzs = orbit(c, k + p)
        g, dg = zs[k + p] - zs[k], dzs[k + p] - dzs[k]
        step = g / dg
        c -= step
        if abs(step) < mp.mpf(10) ** (-(digits + 20)):
            break
    zs, _ = orbit(c, k + p)
    print(f"residual |z_{k+p}-z_{k}| = {mp.nstr(abs(zs[k + p] - zs[k]), 5)}", file=sys.stderr)
    print(f"|z_{k+p-1}-z_{k-1}| = {mp.nstr(abs(zs[k + p - 1] - zs[k - 1]), 5)} (must be >0)", file=sys.stderr)
    print(mp.nstr(c.real, digits, strip_zeros=False))
    print(mp.nstr(c.imag, digits, strip_zeros=False))


if __name__ == "__main__":
    main()
