#include "seq/muradov_k_trapezoid_integral/include/ops_seq.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/split.h>

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

  bool ValidationImpl() override /* NOLINT(readability-make-member-function-const) */ { return (n_ > 0 && a_ <= b_); }

  bool PreProcessingImpl() override /* NOLINT(readability-convert-member-functions-to-static) */ { return true; }

  bool RunImpl() override {
    double h = (b_ - a_) / static_cast<double>(n_);
    struct SumBody {
      double a;
      double h;
      std::function<double(double)> func;
      double sum;
      SumBody(double a, double h, const std::function<double(double)>& f) : a(a), h(h), func(f), sum(0.0) {}
      SumBody(SumBody& other, tbb::split) : a(other.a), h(other.h), func(other.func), sum(0.0) {}
      void operator()(const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i != r.end(); ++i) {
          double x_i = a + (i * h);
          double x_next = a + ((i + 1) * h);
          sum += (func(x_i) + func(x_next)) * 0.5 * h;
        }
      }
      void join(const SumBody& rhs) { sum += rhs.sum; }
    };
    SumBody body(a_, h, func_);
    tbb::parallel_reduce(tbb::blocked_range<int>(0, n_), body);
    result_ = body.sum;
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
