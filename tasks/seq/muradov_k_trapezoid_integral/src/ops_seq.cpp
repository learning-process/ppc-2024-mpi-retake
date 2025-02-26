#include "seq/muradov_k_trapezoid_integral/include/ops_seq.hpp"

#include <cmath>
#include <functional>
#include <memory>

#include "core/task/include/task.hpp"

namespace muradov_k_trapezoid_integral_seq {

namespace {

class IntegrationTask : public ppc::core::Task {
 public:
  IntegrationTask(const std::function<double(double)>& f, double a, double b, int n)
      : ppc::core::Task(std::make_shared<ppc::core::TaskData>()), func_(f), a_(a), b_(b), n_(n) {}

  bool ValidationImpl() override { return (n_ > 0 && a_ <= b_); }

  bool PreProcessingImpl() override { return true; }

  bool RunImpl() override {
    const double h = (b_ - a_) / static_cast<double>(n_);
    double sum = 0.0;

    for (int i = 0; i < n_; ++i) {
      const double x_i = a_ + (i * h);
      const double x_next = a_ + ((i + 1) * h);
      sum += (func_(x_i) + func_(x_next)) * 0.5 * h;
    }

    result_ = sum;
    return true;
  }

  bool PostProcessingImpl() override { return true; }

  [[nodiscard]] double GetResult() const { return result_; }

 private:
  std::function<double(double)> func_;
  double a_, b_;
  int n_;
  double result_ = 0.0;
};

}  // namespace

double GetIntegralTrapezoidalRuleSequential(const std::function<double(double)>& f, double a, double b, int n) {
  auto task = std::make_shared<IntegrationTask>(f, a, b, n);
  if (!task->Validation()) {
    return 0.0;
  }
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
  return task->GetResult();
}

}  // namespace muradov_k_trapezoid_integral_seq