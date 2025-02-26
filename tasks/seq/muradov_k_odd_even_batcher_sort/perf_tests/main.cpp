#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "seq/muradov_k_odd_even_batcher_sort/include/ops_seq.hpp"

namespace mk = muradov_k_odd_even_batcher_sort;

TEST(muradov_k_odd_even_batcher_sort_seq_perf, test_pipeline_run) {
  const int kIterations = 100;
  // Use a large vector (e.g., 256K integers)
  const int n = 256 * 1024;
  std::vector<int> original = mk::random_vector(n);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kIterations; i++) {
    std::vector<int> v = original;  // copy unsorted vector
    mk::odd_even_batcher_sort(v);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_time = end - start;
  double avg_time = total_time.count() / kIterations;
  std::cout << "test_pipeline_run: Average sequential sort time: " << avg_time << " seconds." << std::endl;
  SUCCEED();
}

TEST(muradov_k_odd_even_batcher_sort_seq_perf, test_task_run) {
  const int n = 1024;
  std::vector<int> v = mk::random_vector(n);
  auto start = std::chrono::high_resolution_clock::now();
  mk::odd_even_batcher_sort(v);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "test_task_run: Sequential sort round-trip time: " << elapsed.count() << " seconds." << std::endl;
  SUCCEED();
}
