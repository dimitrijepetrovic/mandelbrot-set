#include <gmpxx.h>

#include <cmath>
#include <stdexcept>

#include "common.hpp"

namespace {

// mpf_class::get_d truncates toward zero; round to nearest like the Rust/Go versions.
double nearest(const mpf_class& x) {
    double d = x.get_d();
    double away = std::nextafter(d, sgn(x) < 0 ? -INFINITY : INFINITY);
    return abs(x - away) < abs(x - d) ? away : d;
}

}  // namespace

// Computes the reference orbit Z_0..Z_max_iter at the zoom centre with enough
// binary precision for the deepest frame, rounding each point to doubles.
Orbit reference_orbit(const std::string& re, const std::string& im, int max_iter, double ln_zoom) {
    auto bits = static_cast<mp_bitcnt_t>(ln_zoom / kLn2 + 64);
    mpf_set_default_prec(bits);
    mpf_class cr, ci;
    if (cr.set_str(re, 10) != 0 || ci.set_str(im, 10) != 0)
        throw std::invalid_argument("invalid centre coordinate");

    Orbit orbit;
    orbit.reserve(2 * (static_cast<size_t>(max_iter) + 1));
    mpf_class zr = 0, zi = 0, zr2, zi2;
    orbit.push_back(0.0);
    orbit.push_back(0.0);
    for (int n = 0; n < max_iter; ++n) {
        zr2 = zr * zr;
        zi2 = zi * zi;
        zi = 2 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        double dr = nearest(zr), di = nearest(zi);
        orbit.push_back(dr);
        orbit.push_back(di);
        if (dr * dr + di * di > kEscapeR2) break;  // reference escaped: shorter orbit
    }
    return orbit;
}
