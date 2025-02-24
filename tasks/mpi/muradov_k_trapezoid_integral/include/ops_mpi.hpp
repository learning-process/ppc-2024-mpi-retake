#ifndef MURADOV_K_TRAPEZOID_INTEGRAL_OPS_MPI_HPP
#define MURADOV_K_TRAPEZOID_INTEGRAL_OPS_MPI_HPP

#include <functional>

namespace muradov_k_trapezoid_integral_mpi {

double GetIntegralTrapezoidalRuleParallel(const std::function<double(double)>& f, double a, double b, int n);

}  // namespace muradov_k_trapezoid_integral_mpi

#endif  // MURADOV_K_TRAPEZOID_INTEGRAL_OPS_MPI_HPP
