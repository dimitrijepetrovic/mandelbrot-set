package main

//go:generate nvcc -O3 -ptx -arch=native -o mandelbrot.ptx ../cuda/mandelbrot.cu

/*
#cgo LDFLAGS: -lcuda
#include <cuda.h>
#include <stdlib.h>

typedef struct {
	CUcontext ctx;
	CUmodule mod;
	CUfunction f64, perturb, perturb_f32;
	CUdeviceptr rgb, ref, ref32; // ref32: the orbit rounded to float, with --gpu-fp32
	size_t bytes;
} gpu_t;

static const char* cu_err(CUresult r) {
	const char* s = "unknown CUDA error";
	cuGetErrorString(r, &s);
	return s;
}

#define TRY(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) return cu_err(r_); } while (0)

static const char* gpu_init(gpu_t* g, const char* ptx, size_t bytes,
                            const double* ref, size_t ref_doubles, int fp32) {
	CUdevice dev;
	TRY(cuInit(0));
	TRY(cuDeviceGet(&dev, 0));
	TRY(cuDevicePrimaryCtxRetain(&g->ctx, dev));
	TRY(cuCtxSetCurrent(g->ctx));
	TRY(cuModuleLoadData(&g->mod, ptx));
	TRY(cuModuleGetFunction(&g->f64, g->mod, "mandel_f64"));
	TRY(cuModuleGetFunction(&g->perturb, g->mod, "mandel_perturb"));
	TRY(cuModuleGetFunction(&g->perturb_f32, g->mod, "mandel_perturb_f32"));
	g->bytes = bytes;
	TRY(cuMemAlloc(&g->rgb, bytes));
	if (ref) {
		TRY(cuMemAlloc(&g->ref, ref_doubles * sizeof(double)));
		TRY(cuMemcpyHtoD(g->ref, ref, ref_doubles * sizeof(double)));
	}
	if (ref && fp32) {
		float* f = malloc(ref_doubles * sizeof(float));
		if (!f) return "out of memory";
		for (size_t i = 0; i < ref_doubles; i++) f[i] = (float)ref[i];
		CUresult r = cuMemAlloc(&g->ref32, ref_doubles * sizeof(float));
		if (r == CUDA_SUCCESS) r = cuMemcpyHtoD(g->ref32, f, ref_doubles * sizeof(float));
		free(f);
		if (r != CUDA_SUCCESS) return cu_err(r);
	}
	return NULL;
}

static const char* gpu_launch(gpu_t* g, CUfunction fn, void** params, int w, int h, void* out) {
	TRY(cuCtxSetCurrent(g->ctx));  // goroutines may move between OS threads
	TRY(cuLaunchKernel(fn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, NULL, params, NULL));
	TRY(cuMemcpyDtoH(out, g->rgb, g->bytes));
	return NULL;
}

static const char* gpu_render_f64(gpu_t* g, int w, int h, double cre, double cim,
                                  double spacing, int max_iter, void* out) {
	void* params[] = {&g->rgb, &w, &h, &cre, &cim, &spacing, &max_iter};
	return gpu_launch(g, g->f64, params, w, h, out);
}

// fp32: use mandel_perturb_f32 (see cuda/mandelbrot.cuh).
static const char* gpu_render_perturb(gpu_t* g, int w, int h, int ref_len, double spacing,
                                      int max_iter, int fp32, void* out) {
	void* params[] = {&g->rgb, &w, &h, fp32 ? &g->ref32 : &g->ref, &ref_len, &spacing, &max_iter};
	return gpu_launch(g, fp32 ? g->perturb_f32 : g->perturb, params, w, h, out);
}

static void gpu_close(gpu_t* g) {
	cuCtxSetCurrent(g->ctx);
	if (g->ref) cuMemFree(g->ref);
	if (g->ref32) cuMemFree(g->ref32);
	if (g->rgb) cuMemFree(g->rgb);
	if (g->mod) cuModuleUnload(g->mod);
	CUdevice dev;
	if (cuDeviceGet(&dev, 0) == CUDA_SUCCESS) cuDevicePrimaryCtxRelease(dev);
}
*/
import "C"

import (
	_ "embed"
	"errors"
	"unsafe"
)

//go:embed mandelbrot.ptx
var ptx string

// f32MinSpacing is 2^-220, MANDEL_F32_MIN_SPACING in cuda/mandelbrot.cuh: below this
// spacing the float32 kernel's scaled deltas would overflow, so deeper frames use the
// float64 kernel.
const f32MinSpacing = 5.9241243523004765e-67

type gpuRenderer struct {
	g             C.gpu_t
	width, height int
	cre, cim      float64
	refLen        int
	fp32          bool
}

func cuErr(msg *C.char) error {
	if msg == nil {
		return nil
	}
	return errors.New("CUDA: " + C.GoString(msg))
}

func newGPURenderer(o Options, cre, cim float64, orbit []float64) (*gpuRenderer, error) {
	r := &gpuRenderer{width: o.Width, height: o.Height, cre: cre, cim: cim, refLen: len(orbit) / 2, fp32: o.GPUFP32}
	cPTX := C.CString(ptx)
	defer C.free(unsafe.Pointer(cPTX))
	var ref *C.double
	if orbit != nil {
		ref = (*C.double)(unsafe.Pointer(&orbit[0]))
	}
	bytes := C.size_t(o.Width * o.Height * 3)
	fp32 := C.int(0)
	if o.GPUFP32 {
		fp32 = 1
	}
	if err := cuErr(C.gpu_init(&r.g, cPTX, bytes, ref, C.size_t(len(orbit)), fp32)); err != nil {
		C.gpu_close(&r.g)
		return nil, err
	}
	return r, nil
}

func (r *gpuRenderer) Render(f Frame, rgb []byte) error {
	out := unsafe.Pointer(&rgb[0])
	w, h := C.int(r.width), C.int(r.height)
	if r.refLen > 0 {
		fp32 := C.int(0)
		if r.fp32 && f.Spacing >= f32MinSpacing {
			fp32 = 1
		}
		return cuErr(C.gpu_render_perturb(&r.g, w, h, C.int(r.refLen), C.double(f.Spacing), C.int(f.MaxIter), fp32, out))
	}
	return cuErr(C.gpu_render_f64(&r.g, w, h, C.double(r.cre), C.double(r.cim), C.double(f.Spacing), C.int(f.MaxIter), out))
}

func (r *gpuRenderer) Close() { C.gpu_close(&r.g) }
