/*
  STREAM Benchmark for CUDA
  Author: Mohammed Sourouri <mohammed.sourouri@ntnu.no>
  Date: December 16, 2013
  Updated: June 5, 2017

  Comment: Code uses kernels from the original
           STREAM Benchmark written by John McCalpin.
           https://www.cs.virginia.edu/stream/
*/

#ifndef KERNELS_H_
#define KERNELS_H_

#include "common.h"
#include "common_gpu.h"

extern "C"
{
void initialize_device_buffer(dim3 thread_blocks, dim3 threads_per_block, real *d_buffer, const real value);
void stream_copy(dim3 thread_blocks, dim3 threads_per_block, real *buf_a, real *buf_b, real *buf_c);
void stream_scale(dim3 thread_blocks, dim3 threads_per_block, real *buf_a, real *buf_b, real *buf_c);
void stream_add(dim3 thread_blocks, dim3 threads_per_block, real *buf_a, real *buf_b, real *buf_c);
void stream_triad(dim3 thread_blocks, dim3 threads_per_block, real *buf_a, real *buf_b, real *buf_c);
}

#endif  // KERNELS_H_

