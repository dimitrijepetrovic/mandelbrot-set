#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "mandelbrot.cuh"

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

class GpuRenderer final : public Renderer {
public:
    GpuRenderer(const Options& o, double cre, double cim, const Orbit* orbit)
        : width_(o.width), height_(o.height), cre_(cre), cim_(cim) {
        bytes_ = static_cast<size_t>(width_) * height_ * 3;
        check(cudaMalloc(&rgb_, bytes_), "cudaMalloc frame");
        if (orbit) {
            ref_len_ = static_cast<int>(orbit->size() / 2);
            check(cudaMalloc(&ref_, orbit->size() * sizeof(double)), "cudaMalloc orbit");
            check(cudaMemcpy(ref_, orbit->data(), orbit->size() * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "upload orbit");
            if (o.gpu_fp32) {
                std::vector<float> f(orbit->begin(), orbit->end());
                check(cudaMalloc(&ref32_, f.size() * sizeof(float)), "cudaMalloc orbit");
                check(cudaMemcpy(ref32_, f.data(), f.size() * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      "upload orbit");
            }
        }
    }

    ~GpuRenderer() override {
        cudaFree(rgb_);
        if (ref_) cudaFree(ref_);
        if (ref32_) cudaFree(ref32_);
    }

    void render(const Frame& f, std::vector<uint8_t>& rgb) override {
        dim3 block(16, 16);
        dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
        if (ref32_ && f.spacing >= MANDEL_F32_MIN_SPACING)
            mandel_perturb_f32<<<grid, block>>>(rgb_, width_, height_, ref32_, ref_len_,
                                                f.spacing, f.max_iter);
        else if (ref_)
            mandel_perturb<<<grid, block>>>(rgb_, width_, height_, ref_, ref_len_, f.spacing,
                                            f.max_iter);
        else
            mandel_f64<<<grid, block>>>(rgb_, width_, height_, cre_, cim_, f.spacing,
                                        f.max_iter);
        check(cudaGetLastError(), "kernel launch");
        check(cudaMemcpy(rgb.data(), rgb_, bytes_, cudaMemcpyDeviceToHost), "download frame");
    }

private:
    int width_, height_;
    double cre_, cim_;
    size_t bytes_ = 0;
    unsigned char* rgb_ = nullptr;
    double* ref_ = nullptr;
    float* ref32_ = nullptr;  // --gpu-fp32: the orbit rounded to float
    int ref_len_ = 0;
};

}  // namespace

std::unique_ptr<Renderer> make_gpu_renderer(const Options& opts, double centre_re,
                                            double centre_im, const Orbit* orbit) {
    return std::make_unique<GpuRenderer>(opts, centre_re, centre_im, orbit);
}
