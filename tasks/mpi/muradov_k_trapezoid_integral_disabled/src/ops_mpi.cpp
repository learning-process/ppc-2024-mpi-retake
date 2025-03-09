#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <functional>
#include <memory>

#include "core/task/include/task.hpp"
namespace muradov_k_trapezoid_integral_mpi {

namespace {

// IntegrationTask implements the trapezoidal integration using Boost.MPI.
class IntegrationTask : public ppc::core::Task {
 public:
  IntegrationTask(const std::function<double(double)>& f, double a, double b, int n)
      : ppc::core::Task(std::make_shared<ppc::core::TaskData>()), func_(f), a_(a), b_(b), n_(n) {}

  bool ValidationImpl() override { return (n_ > 0 && a_ <= b_); }

  bool PreProcessingImpl() override { return true; }

  bool RunImpl() override {
    double h = (b_ - a_) / static_cast<double>(n_);
    boost::mpi::environment env;
    boost::mpi::communicator world;

    const int chunk_size = n_ / world.size();
    const int start = world.rank() * chunk_size;
    const int end = (world.rank() == world.size() - 1) ? n_ : start + chunk_size;

    double local_sum = 0.0;
    for (int i = start; i < end; ++i) {
      const double x_i = a_ + (i * h);
      const double x_next = x_i + h;
      local_sum += (func_(x_i) + func_(x_next)) * 0.5 * h;
    }

    double global_sum = 0.0;
    boost::mpi::reduce(world, local_sum, global_sum, std::plus<>(), 0);
    boost::mpi::broadcast(world, global_sum, 0);

    result_ = global_sum;
    return true;
  }

  bool PostProcessingImpl() override /* NOLINT(readability-convert-member-functions-to-static) */ { return true; }

  [[nodiscard]] double GetResult() const { return result_; }

 private:
  std::function<double(double)> func_;
  double a_, b_;
  int n_;
  double result_ = 0.0;
};

}  // end anonymous namespace

double GetIntegralTrapezoidalRuleParallel(const std::function<double(double)>& f, double a, double b, int n) {
  auto task = std::make_shared<IntegrationTask>(f, a, b, n);
  if (!task->Validation()) {
    return 0.0;
  }
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
  return task->GetResult();
}

}  // namespace muradov_k_trapezoid_integral_mpi