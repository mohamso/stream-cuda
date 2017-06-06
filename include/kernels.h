#ifndef KERNELS_H__
#define KERNELS_H__

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
