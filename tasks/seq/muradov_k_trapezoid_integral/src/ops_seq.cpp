#include "seq/muradov_k_trapezoid_integral/include/ops_seq.hpp"

namespace muradov_k_trapezoid_integral_seq {

double getIntegralTrapezoidalRuleSequential(const std::function<double(double)>& f,
                                            double a, double b, int n) {
    if (n <= 0) {
        return 0.0;
    }

    double sum = 0.0;
    double h = (b - a) / n;

    for (int i = 0; i < n; i++) {
        double x_i = a + i * h;
        double x_next = a + (i + 1) * h;
        sum += (f(x_i) + f(x_next)) * 0.5 * h;
    }
    return sum;
}

}  // namespace muradov_k_trapezoid_integral_seq
