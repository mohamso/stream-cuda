/*
  STREAM Benchmark for CUDA
  Author: Mohammed Sourouri <mohammed.sourouri@ntnu.no>
  Date: December 16, 2013
  Updated: June 5, 2017

  Comment: Code uses kernels from the original
           STREAM Benchmark written by John McCalpin.
           https://www.cs.virginia.edu/stream/
*/

#ifndef COMMON_H_
#define COMMON_H_

typedef double real;
constexpr real k_scale = 3.0;
constexpr size_t num_elems = (2 << 25);

#endif // COMMON_H_

