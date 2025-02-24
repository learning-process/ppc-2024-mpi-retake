#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"
#include <mpi.h>

namespace muradov_k_trapezoid_integral_mpi {

double getIntegralTrapezoidalRuleParallel(const std::function<double(double)>& f,
                                          double a, double b, int n) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double local_sum = 0.0;
    if (n <= 0) {
        return 0.0;
    }

    double h = (b - a) / n;

    // Each process computes a portion of the integral.
    for (int i = rank; i < n; i += size) {
        double x_i = a + i * h;
        double x_next = a + (i + 1) * h;
        local_sum += (f(x_i) + f(x_next)) * 0.5 * h;
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return global_sum;
}

}  // namespace muradov_k_trapezoid_integral_mpi
