// Copyright 2024 Nesterov Alexander
#include "mpi/opolin_d_cg_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

bool opolin_d_cg_method_mpi::CGMethodkMPI::PreProcessingImpl() {
  // init data
  if (world_.rank() == 0) {
    auto* ptr = reinterpret_cast<double*>(task_data->inputs[1]);
    b_.assign(ptr, ptr + n_);

    epsilon_ = *reinterpret_cast<double*>(task_data->inputs[2]);
  }
  return true;
}

bool opolin_d_cg_method_mpi::CGMethodkMPI::ValidationImpl() {
  // check input and output
  if (world_.rank() == 0) {
    if (task_data->inputs_count.empty() || task_data->inputs.size() != 3) return false;

    if (task_data->outputs_count.empty() || task_data->inputs_count[0] != task_data->outputs_count[0] ||
        task_data->outputs.empty())
      return false;

    n_ = task_data->inputs_count[0];
    if (n_ <= 0) return false;

    auto* ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    A_.assign(ptr, ptr + n_ * n_);

    if (!isSimmetric(A_, n_)) return false;

    if (!isPositiveDefinite(A_, n_)) return false;
  }
  return true;
}

bool opolin_d_cg_method_mpi::CGMethodkMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  boost::mpi::broadcast(world_, n_, 0);
  boost::mpi::broadcast(world_, epsilon_, 0);
  size_t local_n = n_ / size + (rank < static_cast<int>(n_ % size) ? 1 : 0);

  std::vector<int> send_counts, displs, send_counts_A, displs_A;
  if (rank == 0) {
    send_counts.resize(size);
    displs.resize(size);
    send_counts_A.resize(size);
    displs_A.resize(size);

    size_t offset = 0, offset_A = 0;
    for (int i = 0; i < size; ++i) {
      size_t rows = n_ / size + (i < static_cast<int>(n_ % size) ? 1 : 0);
      send_counts[i] = static_cast<int>(rows);
      displs[i] = static_cast<int>(offset);
      offset += rows;

      send_counts_A[i] = static_cast<int>(rows * n_);
      displs_A[i] = static_cast<int>(offset_A);
      offset_A += rows * n_;
    }
  }
  std::vector<double> local_A(local_n * n_);
  if (rank == 0) {
    boost::mpi::scatterv(world_, A_.data(), send_counts_A, displs_A, local_A.data(), static_cast<int>(local_n * n_), 0);
  } else {
    boost::mpi::scatterv(world_, local_A.data(), static_cast<int>(local_n * n_), 0);
  }
  std::vector<double> local_b(local_n);
  if (rank == 0) {
    boost::mpi::scatterv(world_, b_.data(), send_counts, displs, local_b.data(), static_cast<int>(local_n), 0);
  } else {
    boost::mpi::scatterv(world_, local_b.data(), static_cast<int>(local_n), 0);
  }
  std::vector<double> local_x(local_n, 0.0);
  std::vector<double> local_r = local_b;
  std::vector<double> local_p = local_r;
  std::vector<double> local_Ap(local_n);
  std::vector<double> full_p(n_);

  double rsquare_prev = 0.0;
  while (true) {
    double local_rsquare = opolin_d_cg_method_mpi::scalarProduct(local_r, local_r);
    double rsquare_k = 0.0;
    boost::mpi::reduce(world_, local_rsquare, rsquare_k, std::plus<double>(), 0);
    boost::mpi::broadcast(world_, rsquare_k, 0);

    rsquare_prev = rsquare_k;
    if (rank == 0) {
      boost::mpi::gatherv(world_, local_p.data(), static_cast<int>(local_n), full_p.data(), send_counts, displs, 0);
    } else {
      boost::mpi::gatherv(world_, local_p.data(), static_cast<int>(local_n), 0);
    }
    boost::mpi::broadcast(world_, full_p, 0);

    for (size_t i = 0; i < local_n; ++i) {
      local_Ap[i] = 0.0;
      for (size_t j = 0; j < n_; ++j) {
        local_Ap[i] += local_A[i * n_ + j] * full_p[j];
      }
    }

    // p^T * A * p
    double local_pAp = opolin_d_cg_method_mpi::scalarProduct(local_p, local_Ap);
    double pAp = 0.0;
    boost::mpi::reduce(world_, local_pAp, pAp, std::plus<double>(), 0);
    boost::mpi::broadcast(world_, pAp, 0);

    // alpha_k
    double alpha_k = rsquare_prev / pAp;

    for (size_t i = 0; i < local_n; ++i) {
      // x_k+1
      local_x[i] += alpha_k * local_p[i];
      // r_k+1
      local_r[i] -= alpha_k * local_Ap[i];
    }

    local_rsquare = opolin_d_cg_method_mpi::scalarProduct(local_r, local_r);
    rsquare_k = 0.0;
    boost::mpi::reduce(world_, local_rsquare, rsquare_k, std::plus<double>(), 0);
    boost::mpi::broadcast(world_, rsquare_k, 0);

    if (sqrt(rsquare_k) < epsilon_) {
      break;
    }
    double beta_k = rsquare_k / rsquare_prev;
    for (size_t i = 0; i < local_n; ++i) {
      local_p[i] = local_r[i] + beta_k * local_p[i];
    }
  }

  x_.resize(n_);
  boost::mpi::gatherv(world_, local_x.data(), static_cast<int>(local_n), x_.data(), send_counts, displs, 0);

  return true;
}

bool opolin_d_cg_method_mpi::CGMethodkMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* out = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(x_, out);
  }
  return true;
}

bool opolin_d_cg_method_mpi::isPositiveDefinite(const std::vector<double>& mat, size_t size) {
  std::vector<double> L(size * size, 0);

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j <= i; j++) {
      double sum = 0;
      if (j == i) {
        for (size_t k = 0; k < j; k++) sum += L[j * size + k] * L[j * size + k];
        double val = mat[j * size + j] - sum;
        if (val <= 0) return false;
        L[j * size + j] = std::sqrt(val);
      } else {
        for (size_t k = 0; k < j; k++) sum += L[i * size + k] * L[j * size + k];
        L[i * size + j] = (mat[i * size + j] - sum) / L[j * size + j];
      }
    }
  }
  return true;
}

bool opolin_d_cg_method_mpi::isSimmetric(const std::vector<double>& mat, size_t size) {
  bool simetric = true;
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if (j != i) {
        if (mat[i * size + j] != mat[j * size + i]) {
          simetric = false;
        }
      }
    }
  }
  return simetric;
}

double opolin_d_cg_method_mpi::scalarProduct(const std::vector<double>& a_, const std::vector<double>& b_) {
  size_t size = a_.size();
  double result = 0.0;
  for (size_t i = 0; i < size; i++) {
    result += a_[i] * b_[i];
  }
  return result;
}

std::vector<double> opolin_d_cg_method_mpi::multiplyVecMat(const std::vector<double>& vec,
                                                           const std::vector<double>& mat) {
  size_t size = vec.size();
  std::vector<double> result(size, 0.0);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      result[i] += mat[i * size + j] * vec[j];
    }
  }
  return result;
}
