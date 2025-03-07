#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace tarakanov_d_global_opt_two_dim_prob_mpi {

class GlobalOptSequential : public ppc::core::Task {
 public:
  explicit GlobalOptSequential(std::shared_ptr<ppc::core::TaskData> task_data_)
      : Task(std::move(task_data_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  friend bool CheckConstraints(double x, double y, int constraint_num, std::vector<double> constraints);
  friend double ComputeFunction(double x, double y, std::vector<double> params);
  friend double GetConstraintsSum(double x, double y, int num, std::vector<double> vec);
  friend bool IsAcceptable (double x, double y, int constraint_num, std::vector<double> constraints);
  double result;
  double delta;
  std::vector<double> bounds;
  std::vector<double> params;
  std::vector<double> constr;
  int constr_num;
  int mode;
};

class GlobalOptMpi : public ppc::core::Task {
 public:
  explicit GlobalOptMpi(std::shared_ptr<ppc::core::TaskData> task_data_)
      : Task(std::move(task_data_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  friend bool CheckConstraints(double x, double y, int constraint_num, std::vector<double> constraints);
  friend double ComputeFunction(double x, double y, std::vector<double> params);
  friend double GetConstraintsSum(double x, double y, int num, std::vector<double> vec);
  friend bool IsAcceptable (double x, double y, int constraint_num, std::vector<double> constraints);
  double result;
  double delta;
  std::vector<double> bounds;
  std::vector<double> params;
  std::vector<double> constr;
  std::vector<double> local_constr;
  int constr_num;
  int mode;

  boost::mpi::communicator world;
  int local_constr_size;
  std::vector<int> is_corret;
};
bool CheckConstraints(double x, double y, int constraint_num, std::vector<double> constraints);
double ComputeFunction(double x, double y, std::vector<double> params);
double GetConstraintsSum(double x, double y, int num, std::vector<double> vec);
bool IsAcceptable (double x, double y, int constraint_num, std::vector<double> constraints);
}  // namespace tarakanov_d_global_opt_two_dim_prob_mpi