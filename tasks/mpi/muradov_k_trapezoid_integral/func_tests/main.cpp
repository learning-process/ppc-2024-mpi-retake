#include <gtest/gtest.h>
#include <mpi.h>
#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"
#include <functional>
#include <cmath>

namespace muradov_k_trapezoid_integral_mpi {

TEST(muradov_k_trapezoid_integral_mpi, SquareFunction) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto f = [](double x) { return x * x; };
    double a = 5.0, b = 10.0;
    int n = 100;

    double global_sum = getIntegralTrapezoidalRuleParallel(f, a, b, n);

    if (rank == 0) {
        // Reference calculation
        double reference_sum = 0.0;
        if (n > 0) {
            double h = (b - a) / n;
            for (int i = 0; i < n; i++) {
                double x_i = a + i * h;
                double x_next = a + (i + 1) * h;
                reference_sum += (f(x_i) + f(x_next)) * 0.5 * h;
            }
        }
        ASSERT_NEAR(reference_sum, global_sum, 1e-6);
    }
}

TEST(muradov_k_trapezoid_integral_mpi, CubeFunction) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto f = [](double x) { return x * x * x; };
    double a = 0.0, b = 6.0;
    int n = 100;

    double global_sum = getIntegralTrapezoidalRuleParallel(f, a, b, n);

    if (rank == 0) {
        double reference_sum = 0.0;
        if (n > 0) {
            double h = (b - a) / n;
            for (int i = 0; i < n; i++) {
                double x_i = a + i * h;
                double x_next = a + (i + 1) * h;
                reference_sum += (f(x_i) + f(x_next)) * 0.5 * h;
            }
        }
        ASSERT_NEAR(reference_sum, global_sum, 1e-6);
    }
}

}  // namespace muradov_k_trapezoid_integral_mpi
