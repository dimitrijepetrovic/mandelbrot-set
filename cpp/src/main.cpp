#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <optional>

#include "common.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) try {
    Options o = parse_options(argc, argv);
    const bool deep = o.precision == Precision::Deep;
    const int frames = std::max(1, static_cast<int>(std::lround(o.fps * o.duration)));
    const double ln_zoom = det_ln(o.zoom);

    // Frame i zooms exponentially from 1x to o.zoom; iterations grow with depth.
    auto frame_at = [&](int i) {
        double t = frames > 1 ? static_cast<double>(i) / (frames - 1) : 0.0;
        return Frame{kViewHeight / (det_exp(t * ln_zoom) * o.height),
                     o.iter_base + static_cast<int>(o.iter_per_decade * t * ln_zoom / kLn10)};
    };

    auto start = Clock::now();
    double ref_s = 0.0;
    std::optional<Orbit> orbit;
    if (deep) {
        auto t0 = Clock::now();
        orbit = reference_orbit(o.centre_re, o.centre_im, frame_at(frames - 1).max_iter, ln_zoom);
        ref_s = seconds_since(t0);
    }
    const double cre = std::stod(o.centre_re), cim = std::stod(o.centre_im);
    auto renderer = o.device == Device::Cpu
                        ? make_cpu_renderer(o, cre, cim, orbit ? &*orbit : nullptr)
                        : make_gpu_renderer(o, cre, cim, orbit ? &*orbit : nullptr);

    std::optional<VideoWriter> video;
    if (!o.no_video) video.emplace(o.output, o.width, o.height, o.fps, o.encoder);

    std::vector<uint8_t> rgb(static_cast<size_t>(o.width) * o.height * 3);
    double render_s = 0.0, write_s = 0.0;
    for (int i = 0; i < frames; ++i) {
        Frame f = frame_at(i);
        auto t0 = Clock::now();
        renderer->render(f, rgb);
        render_s += seconds_since(t0);
        if (video) {
            t0 = Clock::now();
            video->write(rgb);
            write_s += seconds_since(t0);
        }
        std::fprintf(stderr, "\rframe %d/%d  max_iter %d", i + 1, frames, f.max_iter);
    }
    std::fputc('\n', stderr);
    if (video) {
        auto t0 = Clock::now();
        video->finish();
        write_s += seconds_since(t0);
    }
    double total_s = seconds_since(start);

    std::printf(
        "lang=cpp device=%s precision=%s size=%dx%d frames=%d zoom=%.3g ref_s=%.3f "
        "render_s=%.3f write_s=%.3f total_s=%.3f render_fps=%.2f output=%s\n",
        o.device == Device::Cpu ? "cpu" : "gpu", deep ? "deep" : "f64", o.width, o.height,
        frames, o.zoom, ref_s, render_s, write_s, total_s, frames / render_s,
        o.no_video ? "-" : o.output.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nerror: %s\n", e.what());
    return 1;
}
