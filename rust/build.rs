// Compiles the shared CUDA kernels to PTX; the binary embeds it and the CUDA
// driver JIT-compiles it for the installed GPU at start-up.
use std::{env, path::PathBuf, process::Command};

fn main() {
    let src = PathBuf::from("../cuda/mandelbrot.cu");
    let ptx = PathBuf::from(env::var("OUT_DIR").unwrap()).join("mandelbrot.ptx");
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "native".into());
    // CUDA_FMAD=false gives bit-identical CPU/GPU output (see README).
    let fmad = env::var("CUDA_FMAD").unwrap_or_else(|_| "true".into());
    println!("cargo:rerun-if-changed=../cuda/mandelbrot.cu");
    println!("cargo:rerun-if-changed=../cuda/mandelbrot.cuh");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_FMAD");
    let status = Command::new("nvcc")
        .args(["-O3", "-ptx", &format!("-arch={arch}"), &format!("--fmad={fmad}"), "-o"])
        .arg(&ptx)
        .arg(&src)
        .status()
        .expect("failed to run nvcc (is the CUDA toolkit installed?)");
    assert!(status.success(), "nvcc failed to compile {}", src.display());
}
