#include <gtest/gtest.h>

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
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
  double h = (b - a) / n;
  for (int i = 0; i < n; i++) {
    double x_i = a + (i * h);
    double x_next = x_i + h;
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
  double h = (b - a) / n;
  for (int i = 0; i < n; i++) {
    double x_i = a + (i * h);
    double x_next = x_i + h;
    reference_sum += (f(x_i) + f(x_next)) * 0.5 * h;
  }
  ASSERT_NEAR(reference_sum, result, 1e-6);
}

TEST(muradov_k_trapezoid_integral_seq, SinFunction) {
  std::function<double(double)> f = [](double x) { return sin(x); };
  double a = 0.0;
  double b = M_PI;  // Integrate from 0 to pi
  int n = 100;

  double result = GetIntegralTrapezoidalRuleSequential(f, a, b, n);

  double reference_sum = 0.0;
  double h = (b - a) / n;
  for (int i = 0; i < n; i++) {
    double x_i = a + (i * h);
    double x_next = x_i + h;
    reference_sum += (f(x_i) + f(x_next)) * 0.5 * h;
  }
  ASSERT_NEAR(reference_sum, result, 1e-6);
}

TEST(muradov_k_trapezoid_integral_seq, ConstantFunction) {
  // f(x) = 5.0 is constant
  std::function<double(double)> f = [](double) { return 5.0; };
  double a = 1.0;
  double b = 3.0;  // Expected integral: 5 * (3 - 1) = 10
  int n = 10;      // Fewer intervals suffice for a constant function

  double result = GetIntegralTrapezoidalRuleSequential(f, a, b, n);

  double reference_sum = 0.0;
  double h = (b - a) / n;
  for (int i = 0; i < n; i++) {
    double x_i = a + (i * h);
    double x_next = x_i + h;
    reference_sum += (f(x_i) + f(x_next)) * 0.5 * h;
  }
  ASSERT_NEAR(reference_sum, result, 1e-6);
}

TEST(muradov_k_trapezoid_integral_seq, ExponentialFunction) {
  std::function<double(double)> f = [](double x) { return exp(x); };
  double a = 0.0;
  double b = 1.0;
  int n = 100;

  double result = GetIntegralTrapezoidalRuleSequential(f, a, b, n);

  double reference_sum = 0.0;
  double h = (b - a) / n;
  for (int i = 0; i < n; i++) {
    double x_i = a + (i * h);
    double x_next = x_i + h;
    reference_sum += (f(x_i) + f(x_next)) * 0.5 * h;
  }
  ASSERT_NEAR(reference_sum, result, 1e-6);
}

}  // namespace muradov_k_trapezoid_integral_seq
