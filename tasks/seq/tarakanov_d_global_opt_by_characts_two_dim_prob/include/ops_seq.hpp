// @copyright Tarakanov Denis
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tarakanov_d_global_opt_two_dim_prob_seq {

class GlobalOptSequential : public ppc::core::Task {
 public:
  explicit GlobalOptSequential(std::shared_ptr<ppc::core::TaskData> data) : Task(std::move(data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void SaveResult(int mode, double value, double& local_min_x, double& local_min_y, double real_x, double real_y);

  friend bool CheckConstraints(double x, double y, int constraint_num, const std::vector<double>& constraints);
  friend double ComputeFunction(double x, double y, std::vector<double> params);
  friend double GetConstraintsSum(double x, double y, int num, std::vector<double> vec);
  friend bool IsAcceptable(double x, double y, int constraint_num, const std::vector<double>& constraints);

  double delta_;
  std::vector<double> bounds_;
  std::vector<double> params_;
  std::vector<double> constr_;
  int mode_;
  int constr_num_;
  double result_;
};

bool CheckConstraints(double x, double y, int constraint_num, const std::vector<double>& constraints);
double ComputeFunction(double x, double y, std::vector<double> params);
double GetConstraintsSum(double x, double y, int num, std::vector<double> vec);
bool IsAcceptable(double x, double y, int constraint_num, const std::vector<double>& constraints);

}  // namespace tarakanov_d_global_opt_two_dim_prob_seq
