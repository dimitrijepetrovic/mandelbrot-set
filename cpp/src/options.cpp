#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common.hpp"

namespace {

void usage() {
    std::fputs(
        "Usage: mandelbrot [options]\n"
        "  --device cpu|gpu|both     compute device; both = CPU and GPU render alternate\n"
        "                            frames (default cpu)\n"
        "  --precision f64|deep      f64 = plain doubles (zoom <= ~1e12),\n"
        "                            deep = perturbation (zoom <= 1e300) (default f64)\n"
        "  --width N --height N      frame size (default 1920x1080)\n"
        "  --fps N                   frames per second (default 30)\n"
        "  --duration S              video length in seconds (default 20)\n"
        "  --zoom Z                  final magnification (default 1e12 f64, 1e50 deep)\n"
        "  --centre-re S --centre-im S  zoom centre as decimal strings\n"
        "  --iter-base N             max iterations at zoom 1 (default 512)\n"
        "  --iter-per-decade N       extra iterations per 10x zoom (default 128)\n"
        "  --threads N               CPU threads (default: all)\n"
        "  --encoder NAME            ffmpeg video encoder (default libx264)\n"
        "  --output PATH             output file (default mandelbrot_cpp_<device>_<precision>.mp4)\n"
        "  --no-video                render only, skip ffmpeg (pure compute benchmark)\n"
        "  --gpu-fp32                deep mode on the GPU in float32: much faster, not\n"
        "                            bit-identical to the CPU (zoom <= ~1e63)\n",
        stderr);
}

[[noreturn]] void fail(const std::string& msg) {
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    usage();
    std::exit(2);
}

}  // namespace

const char* device_name(Device d) {
    return d == Device::Cpu ? "cpu" : d == Device::Gpu ? "gpu" : "both";
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage();
            std::exit(0);
        }
        if (arg == "--no-video" || arg == "--gpu-fp32") {
            (arg == "--no-video" ? o.no_video : o.gpu_fp32) = true;
            continue;
        }
        if (i + 1 >= argc) fail("missing value for " + arg);
        std::string val = argv[++i];
        try {
            if (arg == "--device") {
                if (val == "cpu") o.device = Device::Cpu;
                else if (val == "gpu") o.device = Device::Gpu;
                else if (val == "both") o.device = Device::Both;
                else fail("--device must be cpu, gpu or both");
            } else if (arg == "--precision") {
                if (val == "f64") o.precision = Precision::F64;
                else if (val == "deep") o.precision = Precision::Deep;
                else fail("--precision must be f64 or deep");
            } else if (arg == "--width") o.width = std::stoi(val);
            else if (arg == "--height") o.height = std::stoi(val);
            else if (arg == "--fps") o.fps = std::stoi(val);
            else if (arg == "--duration") o.duration = std::stod(val);
            else if (arg == "--zoom") o.zoom = std::stod(val);
            else if (arg == "--centre-re") o.centre_re = val;
            else if (arg == "--centre-im") o.centre_im = val;
            else if (arg == "--iter-base") o.iter_base = std::stoi(val);
            else if (arg == "--iter-per-decade") o.iter_per_decade = std::stod(val);
            else if (arg == "--threads") o.threads = std::stoi(val);
            else if (arg == "--encoder") o.encoder = val;
            else if (arg == "--output") o.output = val;
            else fail("unknown option " + arg);
        } catch (const std::logic_error&) {
            fail("invalid value for " + arg + ": " + val);
        }
    }
    if (o.width <= 0 || o.height <= 0 || o.fps <= 0 || o.duration <= 0 || o.iter_base <= 0)
        fail("size, fps, duration and iter-base must be positive");
    if (o.zoom == 0.0) o.zoom = o.precision == Precision::F64 ? 1e12 : 1e50;
    if (o.zoom < 1.0 || o.zoom > kMaxZoom) fail("--zoom must be between 1 and 1e300");
    if (o.gpu_fp32 && (o.precision != Precision::Deep || o.device == Device::Cpu))
        fail("--gpu-fp32 needs --precision deep and --device gpu or both");
    if (o.output.empty())
        o.output = std::string("mandelbrot_cpp_") + device_name(o.device) +
                   "_" + (o.precision == Precision::F64 ? "f64" : "deep") + ".mp4";
    return o;
}
