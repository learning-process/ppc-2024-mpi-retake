#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>

#include "seq/muradov_k_trapezoid_integral/include/ops_seq.hpp"

namespace muradov_k_trapezoid_integral_seq {

TEST(muradov_k_trapezoid_integral_seq, test_task_run) {
  std::function<double(double)> f = [](double x) { return x * sin(x); };

  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  SUCCEED();
}

TEST(muradov_k_trapezoid_integral_seq, test_pipeline_run) {
  std::function<double(double)> f = [](double x) { return exp(x); };

  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  SUCCEED();
}

}  // namespace muradov_k_trapezoid_integral_seq