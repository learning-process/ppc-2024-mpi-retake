#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include "seq/muradov_k_trapezoid_integral/include/ops_seq.hpp"

namespace muradov_k_trapezoid_integral_seq {

TEST(muradov_k_trapezoid_integral_seq, SquareFunction) {
  std::function<double(double)> f = [](double x) { return x * x; };
  double a = 5.0;
  double b = 10.0;
  int n = 100;

  double result = GetIntegralTrapezoidalRuleSequential(f, a, b, n);

  double reference_sum = 0.0;
  double h = (b - a) / static_cast<double>(n);
  for (int i = 0; i < n; i++) {
    double x_i = a + i * h;
    double x_next = a + (i + 1) * h;
    reference_sum += (f(x_i) + f(x_next)) * 0.5 * h;
  }
  ASSERT_NEAR(reference_sum, result, 1e-6);
}

TEST(muradov_k_trapezoid_integral_seq, CubeFunction) {
  std::function<double(double)> f = [](double x) { return x * x * x; };
  double a = 0.0;
  double b = 6.0;
  int n = 100;

  double result = GetIntegralTrapezoidalRuleSequential(f, a, b, n);

  double reference_sum = 0.0;
  double h = (b - a) / static_cast<double>(n);
  for (int i = 0; i < n; i++) {
    double x_i = a + i * h;
    double x_next = a + (i + 1) * h;
    reference_sum += (f(x_i) + f(x_next)) * 0.5 * h;
  }
  ASSERT_NEAR(reference_sum, result, 1e-6);
}

}  // namespace muradov_k_trapezoid_integral_seq
