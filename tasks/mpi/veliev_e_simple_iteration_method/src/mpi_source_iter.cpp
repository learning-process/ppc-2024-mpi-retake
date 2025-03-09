// Copyright 2024 Nesterov Alexander
#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <cmath>
#include <cstring>
#include <vector>

#include "mpi/veliev_e_simple_iteration_method/include/mpi_header_iter.hpp"

namespace veliev_e_simple_iteration_method_mpi {

bool VelievSlaeIterMpi::IsDiagonallyDominant() {
  for (int row = 0; row < matrix_size_; ++row) {
    double diag_value = std::abs(MatrixAt(coeff_matrix_, row, row));
    double row_sum = 0.0;

    for (int col = 0; col < matrix_size_; ++col) {
      if (row != col) {
        row_sum += std::abs(MatrixAt(coeff_matrix_, row, col));
      }
    }

    if (diag_value <= row_sum) {
      return false;
    }
  }
  return true;
}

bool VelievSlaeIterMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs_count[0] != task_data->inputs_count[1] ||
        task_data->inputs_count[0] != task_data->outputs_count[0]) {
      return false;
    }
    matrix_size_ = static_cast<int>(task_data->inputs_count[0]);

    coeff_matrix_.resize(matrix_size_ * matrix_size_);
    std::ranges::copy(reinterpret_cast<double*>(task_data->inputs[0]),
                      reinterpret_cast<double*>(task_data->inputs[0]) + (matrix_size_ * matrix_size_),
                      coeff_matrix_.begin());
    rhs_vector_.resize(matrix_size_);
    std::ranges::copy(reinterpret_cast<double*>(task_data->inputs[1]),
                      reinterpret_cast<double*>(task_data->inputs[1]) + matrix_size_, rhs_vector_.begin());
    return IsDiagonallyDominant();
  }
  return true;
}

bool VelievSlaeIterMpi::PreProcessingImpl() {
  convergence_tolerance_ = 1e-6;
  if (world_.rank() == 0) {
    solution_vector_.resize(matrix_size_);
    std::ranges::copy(reinterpret_cast<double*>(task_data->outputs[0]),
                      reinterpret_cast<double*>(task_data->outputs[0]) + matrix_size_, solution_vector_.begin());

    iteration_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
    free_term_vector_.resize(matrix_size_);
    for (int row = 0; row < matrix_size_; ++row) {
      double diag_value = MatrixAt(coeff_matrix_, row, row);
      if (diag_value == 0.0) {
        return false;
      }
      free_term_vector_[row] = rhs_vector_[row] / diag_value;
      for (int col = 0; col < matrix_size_; ++col) {
        if (row != col) {
          MatrixAt(iteration_matrix_, row, col) = -MatrixAt(coeff_matrix_, row, col) / diag_value;
        }
      }
    }
  }
  return true;
}

bool VelievSlaeIterMpi::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  broadcast(world_, matrix_size_, 0);

  int local_rows = 0;
  int local_displ = 0;
  std::vector<int> rows_per_proc;
  std::vector<int> displs(1, 0);
  std::vector<int> elements_per_proc;
  std::vector<int> element_displs(1, 0);
  if (rank == 0) {
    int base_rows = matrix_size_ / size;
    int extra_rows = matrix_size_ % size;
    if (size != 0) {
      rows_per_proc.resize(size);
      displs.resize(size);
      elements_per_proc.resize(size);
      element_displs.resize(size);
    }

    for (int i = 0; i < size; ++i) {
      rows_per_proc[i] = base_rows + (i < extra_rows ? 1 : 0);
      elements_per_proc[i] = rows_per_proc[i] * matrix_size_;
    }

    displs[0] = 0;
    element_displs[0] = 0;
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + rows_per_proc[i - 1];
      element_displs[i] = element_displs[i - 1] + elements_per_proc[i - 1];
    }
  }

  scatter(world_, rows_per_proc, local_rows, 0);
  scatter(world_, displs, local_displ, 0);

  std::vector<double> local_matrix(local_rows * matrix_size_);
  std::vector<double> local_free_terms(local_rows);
  scatterv(world_, iteration_matrix_.data(), elements_per_proc, element_displs, local_matrix.data(),
           local_rows * matrix_size_, 0);
  scatterv(world_, free_term_vector_.data(), rows_per_proc, displs, local_free_terms.data(), local_rows, 0);
  solution_vector_.resize(matrix_size_);
  broadcast(world_, solution_vector_.data(), matrix_size_, 0);

  std::vector<double> next_solution(matrix_size_, 0.0);
  std::vector<double> local_current(local_rows, 0.0);
  double max_difference = 100.0;
  while (max_difference > convergence_tolerance_) {
    for (int i = 0; i < local_rows; ++i) {
      double sum = 0.0;
      int global_row = local_displ + i;

      for (int j = 0; j < matrix_size_; ++j) {
        if (j != global_row) {
          sum += local_matrix[(i * matrix_size_) + j] * solution_vector_[j];
        }
      }

      local_current[i] = local_free_terms[i] + sum;
    }

    gatherv(world_, local_current.data(), static_cast<int>(local_current.size()), next_solution.data(), rows_per_proc,
            displs, 0);

    if (rank == 0) {
      max_difference = 0.0;
      for (int i = 0; i < matrix_size_; ++i) {
        max_difference = std::max(max_difference, std::abs(next_solution[i] - solution_vector_[i]));
      }
      solution_vector_ = next_solution;
    }

    broadcast(world_, solution_vector_.data(), static_cast<int>(solution_vector_.size()), 0);
    broadcast(world_, max_difference, 0);
  }

  return true;
}

bool VelievSlaeIterMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(solution_vector_, reinterpret_cast<double*>(task_data->outputs[0]));
  }

  return true;
}

}  // namespace veliev_e_simple_iteration_method_mpi
