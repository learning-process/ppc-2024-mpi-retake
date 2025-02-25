#ifndef ERSOZ_B_RECTANGULAR_METHOD_INTEGRATION_OPS_MPI_HPP
#define ERSOZ_B_RECTANGULAR_METHOD_INTEGRATION_OPS_MPI_HPP

#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#include <cstddef>  // size_t için
#include <functional>
#include <stdexcept>  // std::runtime_error için

namespace ersoz_b_rectangular_method_integration_mpi {

double GetIntegralRectangularMethodSequential(const std::function<double(double)>& integrable_function, double a,
                                              double b, size_t count);

double GetIntegralRectangularMethodParallel(const std::function<double(double)>& integrable_function, double a,
                                            double b, size_t count);

}  // namespace ersoz_b_rectangular_method_integration_mpi

#endif  // ERSOZ_B_RECTANGULAR_METHOD_INTEGRATION_OPS_MPI_HPP