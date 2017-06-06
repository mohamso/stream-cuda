#include "kernels.h"

__global__ void initialize(real *buffer, const real value) {
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_elems) { buffer[idx] = value; }
}

__global__ void copy(real *buffer_b, const real *__restrict__ const buffer_a) {
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_elems) { buffer_b[idx] = buffer_a[idx]; }
}

__global__ void scale(real *buffer_c, const real *__restrict__ const buffer_b) {
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_elems) { buffer_c[idx] = k_scale * buffer_b[idx]; }
}

__global__ void add(real *buffer_c, const real *__restrict__ const buffer_a, const real *__restrict__ const buffer_b) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elems) { buffer_c[idx] = buffer_a[idx]+buffer_b[idx]; }
}

__global__ void triad(real *buffer_c, const real *__restrict__ const buffer_a, const real *__restrict__ const buffer_b) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elems) { buffer_c[idx] = buffer_a[idx]+k_scale*buffer_b[idx]; }
}

extern "C" void initialize_device_buffer(dim3 thread_blocks, dim3 threads_per_block, real* d_buffer, const real value) {
   initialize<<<thread_blocks,threads_per_block>>>(d_buffer, value);
}

extern "C" void stream_copy(dim3 thread_blocks, dim3 threads_per_block, real *buf_a, real *buf_b, real *buf_c) {
  copy<<<thread_blocks,threads_per_block>>>(buf_b, buf_a);
}

extern "C" void stream_scale(dim3 thread_blocks, dim3 threads_per_block, real *buf_a, real *buf_b, real *buf_c) {
  scale<<<thread_blocks,threads_per_block>>>(buf_c, buf_b);
}

extern "C" void stream_add(dim3 thread_blocks, dim3 threads_per_block, real *buf_a, real *buf_b, real *buf_c) {
  add<<<thread_blocks,threads_per_block>>>(buf_c, buf_a, buf_b);
}

extern "C" void stream_triad(dim3 thread_blocks, dim3 threads_per_block, real *buf_a, real *buf_b, real *buf_c) {
  triad<<<thread_blocks, threads_per_block>>>(buf_c, buf_a, buf_b);
}