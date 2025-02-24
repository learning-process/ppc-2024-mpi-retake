#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"

namespace muradov_k_trapezoid_integral_mpi {

// Test #1: measure performance of entire Run() (we can call it "test_task_run").
TEST(muradov_k_trapezoid_integral_mpi, test_task_run) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto f = [](double x) { return x * sin(x); };
    double a = 0.0, b = 10.0;
    int n = 10000000;  // Large number for performance measurement

    auto start = std::chrono::high_resolution_clock::now();
    double result = getIntegralTrapezoidalRuleParallel(f, a, b, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    if (rank == 0) {
        std::cout << "[MPI Task Run] Result: " << result
                  << ", Time: " << elapsed.count() << " seconds" << std::endl;
    }
    SUCCEED();
}

// Test #2: measure performance of "pipeline" (we can call it "test_pipeline_run").
TEST(muradov_k_trapezoid_integral_mpi, test_pipeline_run) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto f = [](double x) { return exp(x); };
    double a = -6.0, b = 6.0;
    int n = 10000000;  // Large number for performance measurement

    auto start = std::chrono::high_resolution_clock::now();
    double result = getIntegralTrapezoidalRuleParallel(f, a, b, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    if (rank == 0) {
        std::cout << "[MPI Pipeline Run] Result: " << result
                  << ", Time: " << elapsed.count() << " seconds" << std::endl;
    }
    SUCCEED();
}

}  // namespace muradov_k_trapezoid_integral_mpi
