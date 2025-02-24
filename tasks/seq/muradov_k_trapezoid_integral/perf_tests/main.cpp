#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include "seq/muradov_k_trapezoid_integral/include/ops_seq.hpp"

namespace muradov_k_trapezoid_integral_seq {

// Test #1: measure performance of entire Run() (call it "test_task_run").
TEST(muradov_k_trapezoid_integral_seq, test_task_run) {
    auto f = [](double x) { return x * sin(x); };
    double a = 0.0, b = 10.0;
    int n = 10000000;  // Large number for performance measurement

    auto start = std::chrono::high_resolution_clock::now();
    double result = getIntegralTrapezoidalRuleSequential(f, a, b, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "[Seq Task Run] Result: " << result
              << ", Time: " << elapsed.count() << " seconds" << std::endl;
    SUCCEED();
}

// Test #2: measure performance of "pipeline" (call it "test_pipeline_run").
TEST(muradov_k_trapezoid_integral_seq, test_pipeline_run) {
    auto f = [](double x) { return exp(x); };
    double a = -6.0, b = 6.0;
    int n = 10000000;  // Large number for performance measurement

    auto start = std::chrono::high_resolution_clock::now();
    double result = getIntegralTrapezoidalRuleSequential(f, a, b, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "[Seq Pipeline Run] Result: " << result
              << ", Time: " << elapsed.count() << " seconds" << std::endl;
    SUCCEED();
}

}  // namespace muradov_k_trapezoid_integral_seq
