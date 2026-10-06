#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "common.hpp"

namespace {

// Each worker iterates kLanes pixels of a row at once. One pixel's iterations form a
// serial dependency chain (each step needs the previous z), so a single pixel leaves
// the core mostly idle; independent lanes fill it, and with -march=native the compiler
// packs them into SIMD registers. When a lane finishes, it is coloured and refilled
// with the row's next pixel. Every lane does exactly the scalar operations of the
// one-pixel loop, so the output is unchanged.
constexpr int kLanes = 8;

// Lane state is all doubles (the iteration count too, exact up to 2^53) and the
// per-iteration loops have no branches, so they vectorize.

// f64: z <- z^2 + c from z = 0.
void row_f64(uint8_t* row, int width, double cre, double ci_row, double spacing,
             int max_iter) {
    alignas(64) double zr[kLanes], zi[kLanes], cr[kLanes], ci[kLanes], r2[kLanes], n[kLanes],
        lim[kLanes];
    int px[kLanes];
    int next = 0, active = 0;
    auto load = [&](int k) {
        zr[k] = zi[k] = 0.0;
        n[k] = 0.0;
        if (next < width) {
            double dcr = (next + 0.5 - 0.5 * width) * spacing;
            cr[k] = cre + dcr;
            ci[k] = ci_row;
            lim[k] = max_iter;
            px[k] = next++;
            ++active;
        } else {  // idle lane: c = 0 never escapes
            cr[k] = ci[k] = 0.0;
            lim[k] = INFINITY;
            px[k] = -1;
        }
    };
    for (int k = 0; k < kLanes; ++k) load(k);
    while (active > 0) {
        int any = 0;
#pragma GCC unroll 1  // keep the loop for the vectorizer instead of unrolling it
        for (int k = 0; k < kLanes; ++k) {
            double zr2 = zr[k] * zr[k], zi2 = zi[k] * zi[k];
            double ni = 2.0 * zr[k] * zi[k] + ci[k];
            double nr = zr2 - zi2 + cr[k];
            double rk = nr * nr + ni * ni;
            zr[k] = nr;
            zi[k] = ni;
            r2[k] = rk;
            n[k] += 1.0;
            any |= (rk > kEscapeR2) | (n[k] >= lim[k]);
        }
        if (!any) continue;
        for (int k = 0; k < kLanes; ++k) {
            if (px[k] < 0 || !(r2[k] > kEscapeR2 || n[k] >= lim[k])) continue;
            uint8_t* p = row + 3 * px[k];
            if (r2[k] > kEscapeR2)
                colour(p, static_cast<int>(n[k]), r2[k]);
            else
                p[0] = p[1] = p[2] = 0;
            --active;
            load(k);
        }
    }
}

// deep: perturbation around the reference orbit, dz <- (2Z + dz) dz + dc, with rebasing.
void row_perturb(uint8_t* row, int width, const double* ref, int ref_len, double dci,
                 double spacing, int max_iter) {
    alignas(64) double dzr[kLanes], dzi[kLanes], dcr[kLanes], r2[kLanes], n[kLanes], lim[kLanes];
    alignas(64) long m[kLanes];
    int px[kLanes];
    const long last = ref_len - 1;
    int next = 0, active = 0;
    auto load = [&](int k) {
        dzr[k] = dzi[k] = 0.0;
        n[k] = 0.0;
        m[k] = 0;
        if (next < width) {
            dcr[k] = (next + 0.5 - 0.5 * width) * spacing;
            lim[k] = max_iter;
            px[k] = next++;
            ++active;
        } else {  // idle lane: never finishes
            dcr[k] = 0.0;
            lim[k] = INFINITY;
            px[k] = -1;
        }
    };
    for (int k = 0; k < kLanes; ++k) load(k);
    while (active > 0) {
        // Three passes, so the arithmetic is a straight-line loop the compiler can
        // vectorize: load each lane's Z_m and Z_m+1 (different indices per lane), step,
        // update m.
        alignas(64) double zmr[kLanes], zmi[kLanes], znr[kLanes], zni[kLanes];
        alignas(64) long rebase[kLanes], end[kLanes];
        for (int k = 0; k < kLanes; ++k) {
            const double* z = ref + 2 * m[k];
            zmr[k] = z[0];
            zmi[k] = z[1];
            znr[k] = z[2];
            zni[k] = z[3];
            end[k] = m[k] + 1 == last;
        }
        int any = 0;
#pragma GCC unroll 1  // keep the loop for the vectorizer instead of unrolling it
        for (int k = 0; k < kLanes; ++k) {
            double tr = 2.0 * zmr[k] + dzr[k], ti = 2.0 * zmi[k] + dzi[k];
            double nr = tr * dzr[k] - ti * dzi[k] + dcr[k];
            double ni = tr * dzi[k] + ti * dzr[k] + dci;
            double zr = znr[k] + nr, zi = zni[k] + ni;
            double rk = zr * zr + zi * zi;
            r2[k] = rk;
            n[k] += 1.0;
            any |= (rk > kEscapeR2 && px[k] >= 0) | (n[k] >= lim[k]);
            // Rebase (Zhuoran): restart the reference when z gets closer to 0 than dz.
            rebase[k] = (rk < nr * nr + ni * ni) | end[k];
            dzr[k] = rebase[k] ? zr : nr;
            dzi[k] = rebase[k] ? zi : ni;
        }
        for (int k = 0; k < kLanes; ++k) m[k] = rebase[k] ? 0 : m[k] + 1;
        if (!any) continue;
        for (int k = 0; k < kLanes; ++k) {
            if (px[k] < 0 || !(r2[k] > kEscapeR2 || n[k] >= lim[k])) continue;
            uint8_t* p = row + 3 * px[k];
            if (r2[k] > kEscapeR2)
                colour(p, static_cast<int>(n[k]), r2[k]);
            else
                p[0] = p[1] = p[2] = 0;
            --active;
            load(k);
        }
    }
}

class CpuRenderer final : public Renderer {
public:
    CpuRenderer(const Options& o, double cre, double cim, const Orbit* orbit)
        : width_(o.width), height_(o.height), cre_(cre), cim_(cim), orbit_(orbit),
          threads_(o.threads > 0 ? o.threads : std::max(1u, std::thread::hardware_concurrency())) {}

    void render(const Frame& f, std::vector<uint8_t>& rgb) override {
        // Rows are handed out dynamically: escape times vary a lot across the image.
        std::atomic<int> next_row{0};
        auto worker = [&] {
            for (int y; (y = next_row.fetch_add(1, std::memory_order_relaxed)) < height_;) {
                uint8_t* row = rgb.data() + static_cast<size_t>(y) * width_ * 3;
                double dci = (0.5 * height_ - (y + 0.5)) * f.spacing;
                if (orbit_)
                    row_perturb(row, width_, orbit_->data(), static_cast<int>(orbit_->size() / 2),
                                dci, f.spacing, f.max_iter);
                else
                    row_f64(row, width_, cre_, cim_ + dci, f.spacing, f.max_iter);
            }
        };
        std::vector<std::jthread> pool;
        for (int i = 1; i < threads_; ++i) pool.emplace_back(worker);
        worker();
    }

private:
    int width_, height_;
    double cre_, cim_;
    const Orbit* orbit_;
    int threads_;
};

}  // namespace

std::unique_ptr<Renderer> make_cpu_renderer(const Options& opts, double centre_re,
                                            double centre_im, const Orbit* orbit) {
    return std::make_unique<CpuRenderer>(opts, centre_re, centre_im, orbit);
}
