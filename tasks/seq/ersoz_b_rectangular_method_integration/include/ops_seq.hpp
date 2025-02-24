#ifndef ERSOZ_B_RECTANGULAR_METHOD_INTEGRATION_OPS_SEQ_HPP
#define ERSOZ_B_RECTANGULAR_METHOD_INTEGRATION_OPS_SEQ_HPP

#include <functional>
#include <stdexcept>

namespace ersoz_b_rectangular_method_integration_seq {

double GetIntegralRectangularMethodSequential(const std::function<double(double)>& integrable_function, double a,
                                              double b, size_t count);

}  // namespace ersoz_b_rectangular_method_integration_seq

#endif  // ERSOZ_B_RECTANGULAR_METHOD_INTEGRATION_OPS_SEQ_HPP
