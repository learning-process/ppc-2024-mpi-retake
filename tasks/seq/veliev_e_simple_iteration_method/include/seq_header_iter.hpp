#pragma once
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace veliev_e_simple_iteration_method_seq {

class VelievSlaeIterSeq : public ppc::core::Task {
 public:
  explicit VelievSlaeIterSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int matrix_size_;

  std::vector<double> iteration_matrix_;
  std::vector<double> rhs_vector_;
  std::vector<double> solution_vector_;
  std::vector<double> free_term_vector_;
  std::vector<double> coeff_matrix_;
  double convergence_tolerance_;
  bool IsDiagonallyDominant();

  double& MatrixAt(std::vector<double>& matrix, int row, int col) { return matrix[(row * matrix_size_) + col]; }
};

}  // namespace veliev_e_simple_iteration_method_seq