#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/mpi.hpp>
#include <cmath>
#include <functional>
#include <memory>

#include "modules/core/task/include/task.hpp"

namespace muradov_k_trapezoid_integral_mpi {

namespace {

// IntegrationTask implements the trapezoidal integration using Boost.MPI.
class IntegrationTask : public ppc::core::Task {
 public:
  IntegrationTask(const std::function<double(double)>& f, double a, double b, int n)
      : ppc::core::Task(std::make_shared<ppc::core::TaskData>()),
        func_(boost::bind(f, _1)),
        a_(a),
        b_(b),
        n_(n),
        result_(0.0) {}

  bool ValidationImpl() override { return (n_ > 0 && a_ <= b_); }

  bool PreProcessingImpl() override {
    // No preprocessing needed.
    return true;
  }

  bool RunImpl() override {
    double h = (b_ - a_) / static_cast<double>(n_);
    // Use Boost.MPI for distributed computation.
    boost::mpi::environment env;
    boost::mpi::communicator world;
    int size = world.size();
    int rank = world.rank();
    double local_sum = 0.0;
    for (int i = rank; i < n_; i += size) {
      double x_i = a_ + i * h;
      double x_next = a_ + (i + 1) * h;
      local_sum += (func_(x_i) + func_(x_next)) * 0.5 * h;
    }
    double global_sum = 0.0;
    boost::mpi::reduce(world, local_sum, global_sum, std::plus<double>(), 0);
    boost::mpi::broadcast(world, global_sum, 0);
    result_ = global_sum;
    return true;
  }

  bool PostProcessingImpl() override {
    // No postprocessing needed.
    return true;
  }

  double GetResult() const { return result_; }

 private:
  boost::function<double(double)> func_;
  double a_, b_;
  int n_;
  double result_;
};

}  // end anonymous namespace

double GetIntegralTrapezoidalRuleParallel(const std::function<double(double)>& f, double a, double b, int n) {
  auto task = std::make_shared<IntegrationTask>(f, a, b, n);
  if (!task->Validation()) return 0.0;
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
  return task->GetResult();
}

}  // namespace muradov_k_trapezoid_integral_mpi
