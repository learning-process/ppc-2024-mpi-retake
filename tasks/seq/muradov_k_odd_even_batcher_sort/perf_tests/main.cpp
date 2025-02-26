#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "seq/muradov_k_odd_even_batcher_sort/include/ops_seq.hpp"

namespace mk = muradov_k_odd_even_batcher_sort;

TEST(muradov_k_odd_even_batcher_sort_seq, test_pipeline_run) {
  const int k_iterations = 100;
  // Use a large vector (e.g., 256K integers)
  const int n = 256 * 1024;
  std::vector<int> original = mk::RandomVector(n);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < k_iterations; i++) {
    std::vector<int> v = original;  // copy unsorted vector
    mk::OddEvenBatcherSort(v);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_time = end - start;
  double avg_time = total_time.count() / k_iterations;
  std::cout << "test_pipeline_run: Average sequential sort time: " << avg_time << " seconds.\n";
  SUCCEED();
}

TEST(muradov_k_odd_even_batcher_sort_seq, test_task_run) {
  const int n = 1024;
  std::vector<int> v = mk::RandomVector(n);
  auto start = std::chrono::high_resolution_clock::now();
  mk::OddEvenBatcherSort(v);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "test_task_run: Sequential sort round-trip time: " << elapsed.count() << " seconds.\n";
  SUCCEED();
}
