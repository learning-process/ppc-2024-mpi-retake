#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace somov_i_ribbon_hor_scheme_only_mat_a_mpi {
class RibbonHorSchemeOnlyMatA : public ppc::core::Task {
 public:
  explicit RibbonHorSchemeOnlyMatA(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world_;
  std::vector<int> a_;
  std::vector<int> b_;
  std::vector<int> c_;
  int a_c_ = 0;
  int a_r_ = 0;
  int b_r_ = 0;
  int b_c_ = 0;
};
void LiterallyMult(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c, int a_c, int a_r,
                   int b_c);
}  // namespace somov_i_ribbon_hor_scheme_only_mat_a_mpi