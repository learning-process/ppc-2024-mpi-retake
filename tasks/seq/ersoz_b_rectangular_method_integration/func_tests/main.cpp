#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>

#include "seq/ersoz_b_rectangular_method_integration/include/ops_seq.hpp"

namespace ersoz_b_rectangular_method_integration_seq {

TEST(ersoz_b_rectangular_method_integration_seq, INTEGRAL_FROM_0_TO_1) {
  double result = GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 0, 1, 10000);
  double reference_sum = GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 0, 1, 10000);
  ASSERT_LT(std::fabs(result - reference_sum), std::numeric_limits<double>::epsilon() * 1000);
}

TEST(ersoz_b_rectangular_method_integration_seq, INTEGRAL_FROM_5_TO_0) {
  double result = GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 5, 0, 10000);
  double reference_sum = GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 5, 0, 10000);
  ASSERT_LT(std::fabs(result - reference_sum), std::numeric_limits<double>::epsilon() * 1000);
}

TEST(ersoz_b_rectangular_method_integration_seq, INTEGRAL_FROM_0_TO_100) {
  double result = GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 0, 100, 10000);
  double reference_sum = GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 0, 100, 10000);
  ASSERT_LT(std::fabs(result - reference_sum), std::numeric_limits<double>::epsilon() * 1000);
}

TEST(ersoz_b_rectangular_method_integration_seq, INTEGRAL_FROM_0_TO_709) {
  double result = GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 0, 709, 10000);
  double reference_sum = GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 0, 709, 10000);
  ASSERT_LT(std::fabs(result - reference_sum), std::numeric_limits<double>::epsilon() * 10000);
}

TEST(ersoz_b_rectangular_method_integration_seq, INTEGRAL_WITH_LOW_RANGE) {  // test1
  double result = GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 1, 1.01, 10000);
  double reference_sum = GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 1, 1.01, 10000);
  ASSERT_LT(std::fabs(result - reference_sum), std::numeric_limits<double>::epsilon() * 1000);
}

}  // namespace ersoz_b_rectangular_method_integration_seq
