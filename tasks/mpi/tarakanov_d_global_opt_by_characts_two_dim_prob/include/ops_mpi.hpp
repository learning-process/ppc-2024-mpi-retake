#pragma once

#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tarakanov_d_global_opt_two_dim_prob_mpi {

class GlobalOptSequential : public ppc::core::Task {
 public:
  explicit GlobalOptSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ProcessGridPoints(int int_min_x, int int_max_x, int int_min_y, int int_max_y, int factor, double& local_min_x,
                         double& local_min_y);
  bool IsPointCorrect(double real_x, double real_y);

  friend bool CheckConstraints(double x, double y, int constraint_num, const std::vector<double>& constraints);
  friend double ComputeFunction(double x, double y, std::vector<double> params);
  friend double GetConstraintsSum(double x, double y, int num, std::vector<double> vec);
  friend bool IsAcceptable(double x, double y, int constraint_num, const std::vector<double>& constraints);
  double result_;
  double delta_;
  std::vector<double> bounds_;
  std::vector<double> params_;
  std::vector<double> constr_;
  int constr_num_;
  int mode_;
};

class GlobalOptMpi : public ppc::core::Task {
 public:
  explicit GlobalOptMpi(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ProccessGridPoint(int int_min_x, int int_min_y, int int_max_x, int int_max_y, int factor, double& local_min_x,
                         double& local_min_y);
  void DataDistribution();
  void NewAreaProcess(double& last_result, std::vector<double>& loc_area, double local_min_x, double local_min_y,
                      double& current_step, double accuracy);
  bool CheckCorrect(int sz);
  int ApproveAllConstraints(double real_x, double real_y, int constr_sz);
  void SaveResult(double real_x, double real_y, double value, double& local_min_x, double& local_min_y);

  friend bool CheckConstraints(double x, double y, int constraint_num, const std::vector<double>&);
  friend double ComputeFunction(double x, double y, std::vector<double> params);
  friend double GetConstraintsSum(double x, double y, int num, std::vector<double> vec);
  bool IsAcceptable(double x, double y, int constraint_num, const std::vector<double>& constraints);
  double result_;
  double delta_;
  std::vector<double> bounds_;
  std::vector<double> params_;
  std::vector<double> constr_;
  std::vector<double> local_constr_;
  int constr_num_;
  int mode_;

  boost::mpi::communicator world_;
  int local_constr_size_;
  std::vector<int> is_correct_;
};
bool CheckConstraints(double x, double y, int constraint_num, const std::vector<double>& constraints);
double ComputeFunction(double x, double y, std::vector<double> params);
double GetConstraintsSum(double x, double y, int num, std::vector<double> vec);
bool IsAcceptable(double x, double y, int constraint_num, const std::vector<double>& constraints);
}  // namespace tarakanov_d_global_opt_two_dim_prob_mpi