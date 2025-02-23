#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <cstdlib>
#include <vector>

using namespace std::chrono_literals;

int shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MatrixRank(Matrix matrix, std::vector<double> a) {
  int rank = matrix.cols;
  for (int i = 0; i < matrix.cols; ++i) {
    int j = 0;
    for (j = 0; j < matrix.rows; ++j) {
      if (std::abs(a[(j * matrix.rows) + i]) > 1e-6) {
        break;
      }
    }
    if (j == matrix.rows) {
      --rank;
    } else {
      for (int k = i + 1; k < matrix.cols; ++k) {
        double ml = a[(k * matrix.rows) + i] / a[(i * matrix.rows) + i];
        for (j = i; j < matrix.rows - 1; ++j) {
          a[(k * matrix.rows) + j] -= a[(i * matrix.rows) + j] * ml;
        }
      }
    }
  }
  return rank;
}
double shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::Determinant(Matrix matrix, std::vector<double> a) {
  double det = 1;

  for (int i = 0; i < matrix.cols; ++i) {
    int idx = i;
    for (int k = i + 1; k < matrix.cols; ++k) {
      if (std::abs(a[(k * matrix.rows) + i]) > std::abs(a[(idx * matrix.rows) + i])) {
        idx = k;
      }
    }
    if (std::abs(a[(idx * matrix.rows) + i]) < 1e-6) {
      return 0;
    }
    if (idx != i) {
      for (int j = 0; j < matrix.rows - 1; ++j) {
        double tmp = a[(i * matrix.rows) + j];
        a[(i * matrix.rows) + j] = a[(idx * matrix.rows) + j];
        a[(idx * matrix.rows) + j] = tmp;
      }
      det *= -1;
    }
    det *= a[(i * matrix.rows) + i];
    for (int k = i + 1; k < matrix.cols; ++k) {
      double ml = a[(k * matrix.rows) + i] / a[(i * matrix.rows) + i];
      for (int j = i; j < matrix.rows - 1; ++j) {
        a[(k * matrix.rows) + j] -= a[(i * matrix.rows) + j] * ml;
      }
    }
  }
  return det;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential::PreProcessingImpl() {
  matrix_ = std::vector<double>(task_data->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::ranges::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix_.begin());
  cols_ = static_cast<int>(task_data->inputs_count[1]);
  rows_ = static_cast<int>(task_data->inputs_count[2]);

  res_ = std::vector<double>(cols_ - 1, 0);
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential::ValidationImpl() {
  matrix_ = std::vector<double>(task_data->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::ranges::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix_.begin());
  cols_ = static_cast<int>(task_data->inputs_count[1]);
  rows_ = static_cast<int>(task_data->inputs_count[2]);
  Matrix matrix;
  matrix.cols = cols_;
  matrix.rows = rows_;

  return task_data->inputs_count[0] > 1 && rows_ == cols_ - 1 && Determinant(matrix, matrix_) != 0 &&
         MatrixRank(matrix, matrix_) == rows_;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential::RunImpl() {
  for (int i = 0; i < rows_ - 1; ++i) {
    for (int k = i + 1; k < rows_; ++k) {
      double m = matrix_[(k * cols_) + i] / matrix_[(i * cols_) + i];
      for (int j = i; j < cols_; ++j) {
        matrix_[(k * cols_) + j] -= matrix_[(i * cols_) + j] * m;
      }
    }
  }
  for (int i = rows_ - 1; i >= 0; --i) {
    double sum = matrix_[(i * cols_) + rows_];
    for (int j = i + 1; j < cols_ - 1; ++j) {
      sum -= matrix_[(i * cols_) + j] * res_[j];
    }
    res_[i] = sum / matrix_[(i * cols_) + i];
  }
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential::PostProcessingImpl() {
  auto *this_matrix = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(res_.begin(), res_.end(), this_matrix);
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    matrix_ = std::vector<double>(task_data->inputs_count[0]);
    auto *tmp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    std::ranges::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix_.begin());
    cols_ = static_cast<int>(task_data->inputs_count[1]);
    rows_ = static_cast<int>(task_data->inputs_count[2]);

    res_ = std::vector<double>(cols_ - 1, 0);
  }
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    matrix_ = std::vector<double>(task_data->inputs_count[0]);
    auto *tmp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    std::ranges::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix_.begin());
    cols_ = static_cast<int>(task_data->inputs_count[1]);
    rows_ = static_cast<int>(task_data->inputs_count[2]);
    Matrix matrix;
    matrix.cols = cols_;
    matrix.rows = rows_;

    return task_data->inputs_count[0] > 1 && rows_ == cols_ - 1 && Determinant(matrix, matrix_) != 0 &&
           MatrixRank(matrix, matrix_) == rows_;
  }
  return true;
}

void shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::BroadcastMatrixSize() {
    broadcast(world_, cols_, 0);
    broadcast(world_, rows_, 0);
}

std::vector<int> shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::ComputeRowDistribution() {
    std::vector<int> row_num(world_.size());
    int delta = rows_ / world_.size();
    int remainder = rows_ % world_.size();
    if (world_.rank() < remainder) {
      delta++;
    }
    boost::mpi::gather(world_, delta, row_num, 0);
    return row_num;
}

void shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::DistributeMatrix(const std::vector<int>& row_num) {
    if (world_.rank() == 0) {
        std::vector<double> SendMatrix{delta * cols};
        for (int proc = 1; proc < world_.size(); ++proc) {
            for (int i = 0; i < row_num[proc]; ++i)
                for (int j = 0; j < cols; ++j)
                    SendMatrix[(i * cols) + j] = matrix_[((proc + (world_.size() * i)) * cols_) + j];
            world_.send(proc, 0, SendMatrix.data(), row_num[proc] * cols);
        }
    }
}

void shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::ReceiveMatrix(int delta) {
    local_matrix_.resize(delta * cols);
    if (world_.rank() == 0) {
        for (int i = 0; i < delta; ++i)
            for (int j = 0; j < cols; ++j)
                local_matrix_[(i * cols) + j] = matrix_[(i * cols_ * world_.size()) + j];
    } else {
        world_.recv(0, 0, local_matrix_.data(), delta * cols);
    }
}

void shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::ForwardElimination(std::vector<double>& row) {
    std::vector<double> Pivot(cols);
    int r = 0;
    for (int i = 0; i < rows - 1; ++i) {
        if (i == row[r]) {
            for (int j = 0; j < cols; ++j)
                Pivot[j] = local_matrix_[(r * cols) + j];
            broadcast(world_, Pivot.data(), cols, world_.rank());
            r++;
        } else {
            broadcast(world_, Pivot.data(), cols, i % world_.size());
        }
        for (int k = r; k < delta; ++k) {
            double m = local_matrix_[(k * cols) + i] / Pivot[i];
            for (int j = i; j < cols; ++j)
                local_matrix_[(k * cols) + j] -= Pivot[j] * m;
        }
    }
}

void shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::BackSubstitution(std::vector<double>& row) {
    int r = delta - 1;
    for (int i = rows - 1; i > 0; --i) {
        if (r >= 0 && i == row[r]) {
            local_res_[i] /= local_matrix_[(r * cols) + i];
            broadcast(world_, local_res_[i], world_.rank());
            r--;
        } else {
            broadcast(world_, local_res_[i], i % world_.size());
        }
        if (r >= 0) {
            for (int j = 0; j <= r; ++j)
                local_res_[static_cast<size_t>(row[j])] -= local_matrix_[(j * cols) + i] * local_res_[i];
        }
    }
    if (world_.rank() == 0) {
        local_res_[0] /= local_matrix_[0];
        res = local_res_;
    }
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel::RunImpl() {
    BroadcastMatrixSize();
    std::vector<int> row_num = ComputeRowDistribution();
    DistributeMatrix(row_num);
    ReceiveMatrix(row_num[world_.rank()]);
    
    std::vector<double> row(row_num[world_.rank()]);
    for (int i = 0; i < row.size(); ++i)
        row[i] = world_.rank() + world_.size() * i;
    
    ForwardElimination(row);
    
    local_res_.resize(cols_ - 1, 0);
    BackSubstitution(row);
    return true;
}


bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *this_matrix = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(res_.begin(), res_.end(), this_matrix);
  }
  return true;
}