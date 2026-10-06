# Builds all three implementations into bin/.
#   make                     build everything
#   make CUDA_FMAD=false     bit-identical CPU/GPU output (slower GPU)
#   make CUDA_ARCH=sm_86     target a specific GPU instead of the local one
#   make NATIVE=false        portable CPU code: no -march=native / target-cpu=native,
#                            and Go without GOEXPERIMENT=simd (one pixel at a time)
CUDA_ARCH ?= native
CUDA_FMAD ?= true
NATIVE ?= true
CMAKE_NATIVE := $(if $(filter false,$(NATIVE)),OFF,ON)
# An empty RUSTFLAGS replaces rust/.cargo/config.toml's target-cpu=native.
RUST_ENV := $(if $(filter false,$(NATIVE)),RUSTFLAGS=,)
GO_ENV := $(if $(filter false,$(NATIVE)),,GOEXPERIMENT=simd)
CMAKE_FMAD := $(if $(filter false,$(CUDA_FMAD)),OFF,ON)
CMAKE_ARCH := $(if $(filter sm_%,$(CUDA_ARCH)),$(patsubst sm_%,%,$(CUDA_ARCH)),$(CUDA_ARCH))

.PHONY: all cpp rust go clean FORCE

all: cpp rust go

cpp: | bin
	cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCUDA_FMAD=$(CMAKE_FMAD) -DNATIVE=$(CMAKE_NATIVE) -DCMAKE_CUDA_ARCHITECTURES=$(CMAKE_ARCH)
	cmake --build cpp/build -j
	cp cpp/build/mandelbrot bin/mandelbrot-cpp

rust: | bin
	cd rust && $(RUST_ENV) CUDA_ARCH=$(CUDA_ARCH) CUDA_FMAD=$(CUDA_FMAD) cargo build --release
	cp rust/target/release/mandelbrot bin/mandelbrot-rust

go: | bin
	nvcc -O3 -ptx -arch=$(CUDA_ARCH) --fmad=$(CUDA_FMAD) -o go/mandelbrot.ptx cuda/mandelbrot.cu
	cd go && $(GO_ENV) go vet ./... && $(GO_ENV) go build -o ../bin/mandelbrot-go .

bin:
	mkdir -p bin

clean:
	rm -rf bin cpp/build rust/target go/mandelbrot.ptx
