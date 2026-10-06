#include <atomic>
#include <thread>
#include <vector>

#include "common.hpp"

namespace {

void pixel_f64(uint8_t* px, double cr, double ci, int max_iter) {
    double zr = 0.0, zi = 0.0;
    for (int n = 1; n <= max_iter; ++n) {
        double zr2 = zr * zr, zi2 = zi * zi;
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        double r2 = zr * zr + zi * zi;
        if (r2 > kEscapeR2) return colour(px, n, r2);
    }
    px[0] = px[1] = px[2] = 0;
}

void pixel_perturb(uint8_t* px, const double* ref, int ref_len, double dcr, double dci,
                   int max_iter) {
    double dzr = 0.0, dzi = 0.0;
    int m = 0;
    for (int n = 1; n <= max_iter; ++n) {
        double tr = 2.0 * ref[2 * m] + dzr, ti = 2.0 * ref[2 * m + 1] + dzi;
        double nr = tr * dzr - ti * dzi + dcr;
        dzi = tr * dzi + ti * dzr + dci;
        dzr = nr;
        ++m;
        double zr = ref[2 * m] + dzr, zi = ref[2 * m + 1] + dzi;
        double r2 = zr * zr + zi * zi;
        if (r2 > kEscapeR2) return colour(px, n, r2);
        if (r2 < dzr * dzr + dzi * dzi || m == ref_len - 1) {
            dzr = zr;
            dzi = zi;
            m = 0;
        }
    }
    px[0] = px[1] = px[2] = 0;
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
                for (int x = 0; x < width_; ++x) {
                    double dcr = (x + 0.5 - 0.5 * width_) * f.spacing;
                    if (orbit_)
                        pixel_perturb(row + 3 * x, orbit_->data(),
                                      static_cast<int>(orbit_->size() / 2), dcr, dci, f.max_iter);
                    else
                        pixel_f64(row + 3 * x, cre_ + dcr, cim_ + dci, f.max_iter);
                }
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
