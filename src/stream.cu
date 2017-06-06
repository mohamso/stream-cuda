/*
  STREAM Benchmark for CUDA
  Author: Mohammed Sourouri <mohammed.sourouri@ntnu.no>
  Date: December 16, 2013
  Updated: June 5, 2017

  Comment: Code uses kernels from the original
           STREAM Benchmark written by John McCalpin.
           https://www.cs.virginia.edu/stream/
*/

#include "common.h"

#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <sstream>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>

#include "kernels.h"

template<typename T> void print_row_elem(T t, const int& width) {
  const char whitespace = ' ';
  std::cout << std::left << std::setw(width) << std::setfill(whitespace) << t;
}

void print_separator() {

  std::cout << "+";

  for (unsigned int i = 0; i < 77; i++) {
    std::cout << "-";
  }

  std::cout << "+\n";
}

void print_device_info() {

  int device;

  cudaGetDevice(&device);
  cudaDeviceProp device_properties;
  cudaGetDeviceProperties(&device_properties, device);

  print_row_elem("| Device", 35);
  print_row_elem("ECC Status", 43);
  print_row_elem("|\n", 0);
  print_separator();

  std::cout << "| ";
  print_row_elem(device_properties.name, 33);

  if (device_properties.ECCEnabled == 1) {
    print_row_elem("On", 43);
    print_row_elem("|\n", 0);
  } else {
    print_row_elem("Off", 43);
    print_row_elem("|\n", 0);
  }

}

int main(int argc, char **argv) {

  int num_iters = 0;
  int block_x = 0;

  std::stringstream string_buffer;

  for (int i = 1; i < argc; i++) {
    string_buffer << argv[i] << " ";
  }

  string_buffer >> num_iters;
  string_buffer >> block_x;

  // Device specific memory operations
  real * d_one, *d_two, *d_three;
  size_t d_num_bytes = sizeof(real) * num_elems;

  std::unordered_map<std::string, size_t> size_map {
      {"copy", {2*d_num_bytes}},
      {"scale", {2*d_num_bytes}},
      {"add", {3*d_num_bytes}},
      {"triad", {3*d_num_bytes}}
  };

  cudaMalloc((void **) &d_one, d_num_bytes);
  cudaMalloc((void **) &d_two, d_num_bytes);
  cudaMalloc((void **) &d_three, d_num_bytes);

  // Timer
  float kernel_time = 0.f;
  std::unordered_map<std::string, std::vector<float>> timer_map;

  // CUDA event timers
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  // CUDA launch configuration
  dim3 threads_per_block(block_x);
  dim3 thread_blocks(num_elems/threads_per_block.x);

  if (num_elems % threads_per_block.x != 0) {
    thread_blocks.x+=1;
  }

  // Initialize data on device
  initialize_device_buffer(thread_blocks, threads_per_block, d_one, 2.0);
  initialize_device_buffer(thread_blocks, threads_per_block, d_two, 0.5);
  initialize_device_buffer(thread_blocks, threads_per_block, d_three, 0.5);

  std::unordered_map<std::string, std::function<void(dim3, dim3, real*, real*, real*)>> kernel_func_map {
      {"copy", {stream_copy}},
      {"scale", {stream_scale}},
      {"add", {stream_add}},
      {"triad", {stream_triad}}
  };

  for (const auto& kernel : kernel_func_map) {
    for (int i = 0; i < num_iters+1; i++) {
      cudaEventRecord(start_event, 0);
      kernel.second(thread_blocks, threads_per_block, d_one, d_two, d_three);
      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);
      cudaEventElapsedTime(&kernel_time, start_event, stop_event);
      timer_map[kernel.first].push_back(kernel_time/1000.0f);
    }
    cudaDeviceSynchronize();
    kernel_time = 0.f;
  }

  // Print out benchmark results
  print_separator();
  print_row_elem("|", 30);
  print_row_elem("CUDA STREAM Benchmark", 48);
  print_row_elem("|\n", 0);
  print_separator();

  print_row_elem("| Kernel", 15);
  print_row_elem("Best rate (GB/s)", 20);
  print_row_elem("Avg time", 15);
  print_row_elem("Min time", 19);
  print_row_elem("Max time |\n", 7);
  print_separator();

  for (auto& kernel_time: timer_map) {
    auto min_max = std::minmax_element(kernel_time.second.begin()+1, kernel_time.second.end());
    real avg_time = std::accumulate(kernel_time.second.begin()+1, kernel_time.second.end(), 0.0) / num_iters;
    real bw = 1e-9 * size_map[kernel_time.first] / (*min_max.first);
    std::cout << "| ";
    print_row_elem(kernel_time.first, 13);
    std::cout << std::fixed << std::setprecision(2);
    print_row_elem(bw, 20);
    std::cout << std::fixed << std::setprecision(5);
    print_row_elem(avg_time, 15);
    print_row_elem(*min_max.first, 19);
    print_row_elem(*min_max.second, 8);
    print_row_elem(" |\n", 0);
  }

  print_separator();
  print_row_elem("|", 30);
  print_row_elem("Device Specifications", 47);
  print_row_elem(" |\n", 0);
  print_separator();
  print_device_info();
  print_separator();

  // Free memory on device
  cudaFree(d_one);
  cudaFree(d_two);
  cudaFree(d_three);

  // Destroy events and reset device
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  cudaDeviceReset();

  return 0;

}