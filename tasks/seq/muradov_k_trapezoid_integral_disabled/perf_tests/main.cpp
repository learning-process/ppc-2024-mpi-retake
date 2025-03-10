#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include "seq/muradov_k_trapezoid_integral/include/ops_seq.hpp"

namespace muradov_k_trapezoid_integral_seq {

TEST(muradov_k_trapezoid_integral_seq, test_task_run) {
  std::function<double(double)> f = [](double x) { return x * sin(x); };
  double a = 0.0;
  double b = 10.0;
  int n = 10000000;

  double result = GetIntegralTrapezoidalRuleSequential(f, a, b, n);

  double expected_value = 9.09304866;
  double tolerance = 1e-4;

  ASSERT_NEAR(result, expected_value, tolerance);
}

TEST(muradov_k_trapezoid_integral_seq, test_pipeline_run) {
  std::function<double(double)> f = [](double x) { return exp(x); };
  double a = -6.0;
  double b = 6.0;
  int n = 10000000;

  double result = GetIntegralTrapezoidalRuleSequential(f, a, b, n);

  double expected_value = exp(6) - exp(-6);
  double tolerance = 1e-4;

  ASSERT_NEAR(result, expected_value, tolerance);
}

}  // namespace muradov_k_trapezoid_integral_seq
