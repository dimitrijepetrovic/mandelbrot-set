#include <chrono>
#include <cmath>
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

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
    const Orbit* orb = orbit ? &*orbit : nullptr;
    std::vector<std::unique_ptr<Renderer>> renderers;
    if (o.device != Device::Gpu) renderers.push_back(make_cpu_renderer(o, cre, cim, orb));
    if (o.device != Device::Cpu) renderers.push_back(make_gpu_renderer(o, cre, cim, orb));

    std::optional<VideoWriter> video;
    if (!o.no_video) video.emplace(o.output, o.width, o.height, o.fps, o.encoder);

    // One worker thread per device takes the next frame number from a shared counter,
    // so with --device both the faster device renders more frames. Near the end a device
    // stops if the other would finish all remaining frames before it finished one more
    // (the last frames are the most expensive), so neither waits on the other. The main thread
    // writes finished frames to ffmpeg in order while the workers render ahead, so
    // encoding overlaps rendering. Frame buffers are allocated as needed, so a fast
    // device can run ahead of a slow one's frame (up to max_bufs frames in flight).
    const size_t frame_bytes = static_cast<size_t>(o.width) * o.height * 3;
    const size_t max_bufs = std::clamp<size_t>((size_t{1} << 30) / frame_bytes, 3, 64);
    size_t bufs = 0;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::vector<uint8_t>> free_bufs;
    std::map<int, std::vector<uint8_t>> done;  // rendered, not yet written
    std::vector<int> rendered(renderers.size());
    std::vector<double> last_s(renderers.size());  // each worker's latest frame time
    std::exception_ptr error;
    int next_frame = 0;
    auto yields = [&](size_t w, int i) {  // under mu
        for (size_t v = 0; v < renderers.size(); ++v)
            if (v != w && last_s[v] > 0 && (frames - i) * last_s[v] < last_s[w]) return true;
        return false;
    };

    auto render_start = Clock::now();
    {
        std::vector<std::jthread> workers;
        for (size_t w = 0; w < renderers.size(); ++w) {
            workers.emplace_back([&, w] {
                try {
                    for (;;) {
                        int i;
                        std::vector<uint8_t> rgb;
                        {
                            std::unique_lock lock(mu);
                            cv.wait(lock, [&] { return !free_bufs.empty() || bufs < max_bufs || error; });
                            if (error || next_frame >= frames || yields(w, next_frame)) return;
                            i = next_frame++;
                            if (free_bufs.empty()) {
                                ++bufs;
                                rgb.resize(frame_bytes);
                            } else {
                                rgb = std::move(free_bufs.back());
                                free_bufs.pop_back();
                            }
                        }
                        auto t0 = Clock::now();
                        renderers[w]->render(frame_at(i), rgb);
                        std::lock_guard lock(mu);
                        last_s[w] = seconds_since(t0);
                        done.emplace(i, std::move(rgb));
                        ++rendered[w];
                        cv.notify_all();
                    }
                } catch (...) {
                    std::lock_guard lock(mu);
                    if (!error) error = std::current_exception();
                    cv.notify_all();
                }
            });
        }
        try {
            for (int i = 0; i < frames; ++i) {
                std::vector<uint8_t> rgb;
                {
                    std::unique_lock lock(mu);
                    cv.wait(lock, [&] { return done.contains(i) || error; });
                    if (error) break;
                    rgb = std::move(done.extract(i).mapped());
                }
                if (video) video->write(rgb);
                std::fprintf(stderr, "\rframe %d/%d  max_iter %d", i + 1, frames,
                             frame_at(i).max_iter);
                std::lock_guard lock(mu);
                free_bufs.push_back(std::move(rgb));
                cv.notify_all();
            }
        } catch (...) {
            std::lock_guard lock(mu);
            if (!error) error = std::current_exception();
            cv.notify_all();
        }
    }  // joins the workers
    std::fputc('\n', stderr);
    if (error) std::rethrow_exception(error);
    // render_s: until the last frame was rendered and written (encoding overlaps it);
    // write_s: flushing ffmpeg after that.
    double render_s = seconds_since(render_start), write_s = 0.0;
    if (video) {
        auto t0 = Clock::now();
        video->finish();
        write_s = seconds_since(t0);
    }
    if (o.device == Device::Both)
        std::fprintf(stderr, "frames rendered: cpu %d, gpu %d\n", rendered[0], rendered[1]);
    double total_s = seconds_since(start);

    std::printf(
        "lang=cpp device=%s precision=%s size=%dx%d frames=%d zoom=%.3g ref_s=%.3f "
        "render_s=%.3f write_s=%.3f total_s=%.3f render_fps=%.2f output=%s\n",
        device_name(o.device), deep ? (o.gpu_fp32 ? "deep-fp32" : "deep") : "f64", o.width, o.height,
        frames, o.zoom, ref_s, render_s, write_s, total_s, frames / render_s,
        o.no_video ? "-" : o.output.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nerror: %s\n", e.what());
    return 1;
}
