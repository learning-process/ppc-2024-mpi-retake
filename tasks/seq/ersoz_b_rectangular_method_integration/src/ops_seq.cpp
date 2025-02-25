#include "seq/ersoz_b_rectangular_method_integration/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>

namespace ersoz_b_rectangular_method_integration_seq {

double GetIntegralRectangularMethodSequential(const std::function<double(double)>& integrable_function, double a,
                                              double b, size_t count) {
  if (count == 0) {
    throw std::runtime_error("Zero rectangles count");
  }
  double result = 0.0;
  double delta = (b - a) / static_cast<double>(count);
  for (size_t i = 0; i < count; i++) {
    result += integrable_function(a + (static_cast<double>(i) * delta));
  }
  result *= delta;
  return result;
}

}  // namespace ersoz_b_rectangular_method_integration_seq
