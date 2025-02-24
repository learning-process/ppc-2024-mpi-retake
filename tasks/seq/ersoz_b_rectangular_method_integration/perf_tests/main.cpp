#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>

#include "seq/ersoz_b_rectangular_method_integration/include/ops_seq.hpp"

namespace ersoz_b_rectangular_method_integration_seq {

TEST(ersoz_b_rectangular_method_integration_seq, test_task_run) {
  std::function<double(double)> f = [](double x) { return cos(x); };
  double a = 0.0;
  double b = 100.0;
  size_t count = 10000000;

  auto start = std::chrono::high_resolution_clock::now();
  double result = GetIntegralRectangularMethodSequential(f, a, b, count);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "[Seq Task Run] Result: " << result << ", Time: " << elapsed.count() << " seconds\n";
  SUCCEED();
}

TEST(ersoz_b_rectangular_method_integration_seq, test_pipeline_run) {
  std::function<double(double)> f = [](double x) { return cos(x); };
  double a = 0.0;
  double b = 1000.0;
  size_t count = 10000000;

  auto start = std::chrono::high_resolution_clock::now();
  double result = GetIntegralRectangularMethodSequential(f, a, b, count);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "[Seq Pipeline Run] Result: " << result << ", Time: " << elapsed.count() << " seconds\n";
  SUCCEED();
}

}  // namespace ersoz_b_rectangular_method_integration_seq
