#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Default zoom centre: Misiurewicz point M(23,2), refined to 340 digits with
// tools/misiurewicz.py. Boundary points like this have detail at every depth.
inline constexpr const char* kDefaultCentreRe =
    "-0.7766105925997018565640395025529947493281703214398660009692436578520468609045987087869614280320504142913293744385181191871713086676213649222522743166627219099561021923842471419613761736622513496509490647163308646781007447098756656445264983237348114769587849363429595360430698257987637989567458986520611392894629900349384819163368360786901345";
inline constexpr const char* kDefaultCentreIm =
    "0.1346089616750281660567372702330578095411874962204036173399792327554399692618243468637467375372086757134654215617641155866771648880044865998923176610765412471665957254547656437166294650322153138263014932943834549306648786205329836392513593309204091729193444399806460589188944478163945916072585799737463142109726240996892519451085765977598015";

// Must match cuda/mandelbrot.cu and the Rust/Go implementations.
inline constexpr double kEscapeR2 = 65536.0;
inline constexpr double kColourDensity = 0.015;
inline constexpr double kTwoPi = 6.283185307179586;
inline constexpr double kLn2 = 0.6931471805599453;
inline constexpr double kLn10 = 2.302585092994046;
inline constexpr double kSqrtHalf = 0.7071067811865476;
inline constexpr double kViewHeight = 3.0;  // complex-plane height of frame 0
inline constexpr double kMaxZoom = 1e300;   // f64 deltas underflow beyond this

enum class Device { Cpu, Gpu, Both };
enum class Precision { F64, Deep };

struct Options {
    Device device = Device::Cpu;
    Precision precision = Precision::F64;
    int width = 1920;
    int height = 1080;
    int fps = 30;
    double duration = 20.0;
    double zoom = 0.0;  // 0 = default for the precision
    std::string centre_re = kDefaultCentreRe;
    std::string centre_im = kDefaultCentreIm;
    int iter_base = 512;
    double iter_per_decade = 128.0;
    int threads = 0;  // 0 = all hardware threads
    std::string encoder = "libx264";
    std::string output;  // empty = default name
    bool no_video = false;
    bool gpu_fp32 = false;  // deep on the GPU in float32 (mandel_perturb_f32)
};

Options parse_options(int argc, char** argv);
const char* device_name(Device d);

// exp/ln built only from correctly rounded IEEE operations and exact power-of-two
// scaling, so all three languages compute bit-identical frame parameters
// (libm pow/log/exp differ in the last bit between Go and C).
inline double det_exp(double x) {
    double k = std::round(x / kLn2);
    double r = x - k * kLn2;
    double s = 1.0;
    for (int i = 24; i >= 1; --i) s = 1.0 + s * r / i;
    return std::ldexp(s, static_cast<int>(k));
}

inline double det_ln(double x) {
    int e;
    double m = std::frexp(x, &e);  // x = m * 2^e, m in [0.5, 1)
    if (m < kSqrtHalf) {
        m *= 2.0;
        --e;
    }
    double s = (m - 1.0) / (m + 1.0), s2 = s * s, sum = 0.0;
    for (int i = 41; i >= 1; i -= 2) sum = sum * s2 + 1.0 / i;
    return 2.0 * s * sum + e * kLn2;
}

// Per-frame view parameters.
struct Frame {
    double spacing;  // complex-plane distance between adjacent pixels
    int max_iter;
};

// Reference orbit as interleaved (re, im) doubles; ref[0] == 0.
using Orbit = std::vector<double>;
Orbit reference_orbit(const std::string& re, const std::string& im, int max_iter, double ln_zoom);

class Renderer {
public:
    virtual ~Renderer() = default;
    virtual void render(const Frame& frame, std::vector<uint8_t>& rgb) = 0;
};

std::unique_ptr<Renderer> make_cpu_renderer(const Options& opts, double centre_re,
                                            double centre_im, const Orbit* orbit);
std::unique_ptr<Renderer> make_gpu_renderer(const Options& opts, double centre_re,
                                            double centre_im, const Orbit* orbit);

inline void colour(uint8_t* px, int n, double r2) {
    double log_zn = 0.5 * std::log(r2);
    double nu = n + 1.0 - std::log(log_zn / kLn2) / kLn2;
    double t = nu * kColourDensity;
    px[0] = static_cast<uint8_t>((0.5 + 0.5 * std::cos(kTwoPi * (t + 0.00))) * 255.0 + 0.5);
    px[1] = static_cast<uint8_t>((0.5 + 0.5 * std::cos(kTwoPi * (t + 0.15))) * 255.0 + 0.5);
    px[2] = static_cast<uint8_t>((0.5 + 0.5 * std::cos(kTwoPi * (t + 0.30))) * 255.0 + 0.5);
}

class VideoWriter {
public:
    VideoWriter(const std::string& path, int width, int height, int fps,
                const std::string& encoder);
    ~VideoWriter();
    void write(const std::vector<uint8_t>& rgb);
    void finish();

private:
    int fd_ = -1;
    int pid_ = -1;
};
