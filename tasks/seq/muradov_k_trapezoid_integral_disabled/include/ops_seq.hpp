#ifndef MURADOV_K_TRAPEZOID_INTEGRAL_OPS_SEQ_HPP
#define MURADOV_K_TRAPEZOID_INTEGRAL_OPS_SEQ_HPP

#include <functional>

namespace muradov_k_trapezoid_integral_seq {

double GetIntegralTrapezoidalRuleSequential(const std::function<double(double)>& f, double a, double b, int n);

}  // namespace muradov_k_trapezoid_integral_seq

#endif  // MURADOV_K_TRAPEZOID_INTEGRAL_OPS_SEQ_HPP