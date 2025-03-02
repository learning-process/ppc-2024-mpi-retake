// @copyright Tarakanov Denis
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
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
  friend bool CheckConstraints(double x, double y, int constraint_num, std::vector<double> constraints);
  friend double ComputeFunction(double x, double y, std::vector<double> params);
  friend double GetConstraintsSum(double x, double y, int num, std::vector<double> vec);
  friend bool IsAcceptable (double x, double y, int constraint_num, std::vector<double> constraints);
  
  double step;
  std::vector<double> bounds;
  std::vector<double> func_params;
  std::vector<double> constraints;
  int mode;
  int num_constraints;
  double result;
};

bool CheckConstraints(double x, double y, int constraint_num, std::vector<double> constraints);
double ComputeFunction(double x, double y, std::vector<double> params);
double GetConstraintsSum(double x, double y, int num, std::vector<double> vec);
bool IsAcceptable (double x, double y, int constraint_num, std::vector<double> constraints);

}  // namespace tarakanov_d_global_opt_two_dim_prob_mpi
