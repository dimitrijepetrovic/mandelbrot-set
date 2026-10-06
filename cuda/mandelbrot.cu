#include "mandelbrot.cuh"

// Must match the constants in every CPU implementation (see README "Algorithm").
#define ESCAPE_R2 65536.0
#define COLOUR_DENSITY 0.015
#define TWO_PI 6.283185307179586
#define LN2 0.6931471805599453

__device__ static void colour(unsigned char* px, int n, double r2) {
    double log_zn = 0.5 * log(r2);
    double nu = n + 1.0 - log(log_zn / LN2) / LN2;
    double t = nu * COLOUR_DENSITY;
    px[0] = (unsigned char)((0.5 + 0.5 * cos(TWO_PI * (t + 0.00))) * 255.0 + 0.5);
    px[1] = (unsigned char)((0.5 + 0.5 * cos(TWO_PI * (t + 0.15))) * 255.0 + 0.5);
    px[2] = (unsigned char)((0.5 + 0.5 * cos(TWO_PI * (t + 0.30))) * 255.0 + 0.5);
}

__device__ static void black(unsigned char* px) { px[0] = px[1] = px[2] = 0; }

extern "C" __global__ void mandel_f64(unsigned char* rgb, int width, int height,
                                      double centre_re, double centre_im,
                                      double spacing, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    unsigned char* px = rgb + 3 * ((size_t)y * width + x);

    double cr = centre_re + (x + 0.5 - 0.5 * width) * spacing;
    double ci = centre_im + (0.5 * height - (y + 0.5)) * spacing;
    double zr = 0.0, zi = 0.0;
    for (int n = 1; n <= max_iter; ++n) {
        double zr2 = zr * zr, zi2 = zi * zi;
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        double r2 = zr * zr + zi * zi;
        if (r2 > ESCAPE_R2) { colour(px, n, r2); return; }
    }
    black(px);
}

extern "C" __global__ void mandel_perturb(unsigned char* rgb, int width, int height,
                                          const double* ref, int ref_len,
                                          double spacing, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    unsigned char* px = rgb + 3 * ((size_t)y * width + x);

    double dcr = (x + 0.5 - 0.5 * width) * spacing;
    double dci = (0.5 * height - (y + 0.5)) * spacing;
    double dzr = 0.0, dzi = 0.0;
    int m = 0;
    for (int n = 1; n <= max_iter; ++n) {
        // dz' = (2Z + dz) dz + dc
        double tr = 2.0 * ref[2 * m] + dzr, ti = 2.0 * ref[2 * m + 1] + dzi;
        double nr = tr * dzr - ti * dzi + dcr;
        dzi = tr * dzi + ti * dzr + dci;
        dzr = nr;
        ++m;
        double zr = ref[2 * m] + dzr, zi = ref[2 * m + 1] + dzi;
        double r2 = zr * zr + zi * zi;
        if (r2 > ESCAPE_R2) { colour(px, n, r2); return; }
        if (r2 < dzr * dzr + dzi * dzi || m == ref_len - 1) {
            dzr = zr; dzi = zi; m = 0;  // rebase onto the start of the orbit
        }
    }
    black(px);
}
