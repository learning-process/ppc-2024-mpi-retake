#include "mpi/agafeev_s_strassen_alg/include/strassen_mpi.hpp"

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <functional>
#include <vector>

namespace {
bool IsPowerOfTwo(unsigned int n) { return (n != 0) && ((n & (n - 1)) == 0); }
} // namespace

namespace agafeev_s_strassen_alg_mpi {

std::vector<double> AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

std::vector<double> SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b, int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] - b[i];
  }
  return c;
}

void SplitMatrix(const std::vector<double>& a, std::vector<double>& a11, std::vector<double>& a12,
                 std::vector<double>& a21, std::vector<double>& a22, int n) {
  int half = n / 2;
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      a11[(i * half) + j] = a[(i * n) + j];
      a12[(i * half) + j] = a[(i * n) + (j + half)];
      a21[(i * half) + j] = a[((i + half) * n) + j];
      a22[(i * half) + j] = a[((i + half) * n) + (j + half)];
    }
  }
}

std::vector<double> MergeMatrices(const std::vector<double>& a11, const std::vector<double>& a12,
                                  const std::vector<double>& a21, const std::vector<double>& a22, int n) {
  int half = n / 2;
  std::vector<double> a(n * n);
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      a[(i * n) + j] = a11[(i * half) + j];
      a[(i * n) + (j + half)] = a12[(i * half) + j];
      a[((i + half) * n) + j] = a21[(i * half) + j];
      a[((i + half) * n) + (j + half)] = a22[(i * half) + j];
    }
  }
  return a;
}

std::vector<double> StrassenMultiplySeq(const std::vector<double>& a, const std::vector<double>& b, int n) {
  if (n == 1) {
    return {a[0] * b[0]};
  }

  int half = n / 2;

  std::vector<double> a11(half * half);
  std::vector<double> a12(half * half);
  std::vector<double> a21(half * half);
  std::vector<double> a22(half * half);
  std::vector<double> b11(half * half);
  std::vector<double> b12(half * half);
  std::vector<double> b21(half * half);
  std::vector<double> b22(half * half);
  SplitMatrix(a, a11, a12, a21, a22, n);
  SplitMatrix(b, b11, b12, b21, b22, n);

  auto p1 =
      agafeev_s_strassen_alg_mpi::StrassenMultiplySeq(AddMatrices(a11, a22, half), AddMatrices(b11, b22, half), half);
  auto p2 = agafeev_s_strassen_alg_mpi::StrassenMultiplySeq(AddMatrices(a21, a22, half), b11, half);
  auto p3 = agafeev_s_strassen_alg_mpi::StrassenMultiplySeq(a11, SubtractMatrices(b12, b22, half), half);
  auto p4 = agafeev_s_strassen_alg_mpi::StrassenMultiplySeq(a22, SubtractMatrices(b21, b11, half), half);
  auto p5 = agafeev_s_strassen_alg_mpi::StrassenMultiplySeq(AddMatrices(a11, a12, half), b22, half);
  auto p6 = agafeev_s_strassen_alg_mpi::StrassenMultiplySeq(SubtractMatrices(a21, a11, half),
                                                            AddMatrices(b11, b12, half), half);
  auto p7 = agafeev_s_strassen_alg_mpi::StrassenMultiplySeq(SubtractMatrices(a12, a22, half),
                                                            AddMatrices(b21, b22, half), half);

  auto c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half), p5, half), p7, half);
  auto c12 = AddMatrices(p3, p5, half);
  auto c21 = AddMatrices(p2, p4, half);
  auto c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half), p2, half), p6, half);

  return MergeMatrices(c11, c12, c21, c22, n);
}

std::vector<double> MultiplMatrixMpi::StrassenMultiplyMpi(const std::vector<double>& a, const std::vector<double>& b,
                                                          int n) {
  if (n == 1) {
    return {a[0] * b[0]};
  }

  if (world_.rank() > 6) {
    world_.split(1);
    return {};
  }

  boost::mpi::communicator comm = world_.split(0);

  int rank = comm.rank();
  int size = comm.size();
  boost::mpi::broadcast(comm, n, 0);
  std::vector<double> a_loc(n * n, 0.0);
  std::vector<double> b_loc(n * n, 0.0);
  if (rank == 0) {
    for (int i = 0; i < n * n; i++) {
      a_loc[i] = a[i];
      b_loc[i] = b[i];
    }
  }
  boost::mpi::broadcast(comm, a_loc.data(), n * n, 0);
  boost::mpi::broadcast(comm, b_loc.data(), n * n, 0);

  int half = n / 2;
  std::vector<double> a11(half * half);
  std::vector<double> a12(half * half);
  std::vector<double> a21(half * half);
  std::vector<double> a22(half * half);
  std::vector<double> b11(half * half);
  std::vector<double> b12(half * half);
  std::vector<double> b21(half * half);
  std::vector<double> b22(half * half);
  SplitMatrix(a_loc, a11, a12, a21, a22, n);
  SplitMatrix(b_loc, b11, b12, b21, b22, n);
  std::vector<std::vector<double>> p(7, std::vector<double>(half * half, 0.0));

  for (int i = rank; i < 7; i += size) {
    switch (i) {
      case 0:
        p[0] = StrassenMultiplySeq(AddMatrices(a11, a22, half), AddMatrices(a11, a22, half), half);
        break;
      case 1:
        p[1] = StrassenMultiplySeq(AddMatrices(a21, a22, half), a11, half);
        break;
      case 2:
        p[2] = StrassenMultiplySeq(a11, SubtractMatrices(b12, b22, half), half);
        break;
      case 3:
        p[3] = StrassenMultiplySeq(a22, SubtractMatrices(b21, b11, half), half);
        break;
      case 4:
        p[4] = StrassenMultiplySeq(AddMatrices(a11, a12, half), b22, half);
        break;
      case 5:
        p[5] = StrassenMultiplySeq(SubtractMatrices(a21, a11, half), AddMatrices(b11, b12, half), half);
        break;
      case 6:
        p[6] = StrassenMultiplySeq(SubtractMatrices(a12, a22, half), AddMatrices(b21, b22, half), half);
        break;
    }
  }

  std::vector<double> global_p(7 * half * half, 0.0);
  for (int i = 0; i < 7; ++i) {
    boost::mpi::reduce(comm, p[i].data(), half * half, global_p.data() + (i * half * half), std::plus(), 0);
  }

  if (world_.rank() == 0) {
    std::vector<std::vector<double>> p_reconstructed(7);
    for (int i = 0; i < 7; ++i) {
      p_reconstructed[i] =
          std::vector<double>(global_p.begin() + i * half * half, global_p.begin() + (i + 1) * half * half);
    }

    auto c11 = AddMatrices(
        SubtractMatrices(AddMatrices(p_reconstructed[0], p_reconstructed[3], half), p_reconstructed[4], half),
        p_reconstructed[6], half);
    auto c12 = AddMatrices(p_reconstructed[2], p_reconstructed[4], half);
    auto c21 = AddMatrices(p_reconstructed[1], p_reconstructed[3], half);
    auto c22 = AddMatrices(
        SubtractMatrices(AddMatrices(p_reconstructed[0], p_reconstructed[2], half), p_reconstructed[1], half),
        p_reconstructed[5], half);

    return MergeMatrices(c11, c12, c21, c22, n);
  }
  return {};
}

bool MultiplMatrixSequental::PreProcessingImpl() {
  first_input_.clear();
  second_input_.clear();
  auto* temp_ptr1 = reinterpret_cast<double*>(task_data->inputs[0]);
  first_input_.insert(first_input_.begin(), temp_ptr1,
                      temp_ptr1 + (task_data->inputs_count[0] * task_data->inputs_count[1]));
  auto* temp_ptr2 = reinterpret_cast<double*>(task_data->inputs[1]);
  second_input_.insert(second_input_.begin(), temp_ptr2,
                       temp_ptr2 + (task_data->inputs_count[2] * task_data->inputs_count[3]));
  size_ = task_data->inputs_count[0];

  return true;
}

bool MultiplMatrixSequental::ValidationImpl() {
  return (task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[1] &&
          IsPowerOfTwo(task_data->inputs_count[0]) && IsPowerOfTwo(task_data->inputs_count[1]) &&
          IsPowerOfTwo(task_data->inputs_count[2]) && IsPowerOfTwo(task_data->inputs_count[3]));
}

bool MultiplMatrixSequental::RunImpl() {
  result_ = StrassenMultiplySeq(first_input_, second_input_, size_);

  return true;
}

bool MultiplMatrixSequental::PostProcessingImpl() {
  for (unsigned int i = 0; i < task_data->outputs_count[0]; i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = result_[i];
  }
  return true;
}

bool MultiplMatrixMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    first_input_.clear();
    second_input_.clear();
    auto* temp_ptr1 = reinterpret_cast<double*>(task_data->inputs[0]);
    first_input_.insert(first_input_.begin(), temp_ptr1,
                        temp_ptr1 + (task_data->inputs_count[0] * task_data->inputs_count[1]));
    auto* temp_ptr2 = reinterpret_cast<double*>(task_data->inputs[1]);
    second_input_.insert(second_input_.begin(), temp_ptr2,
                         temp_ptr2 + (task_data->inputs_count[2] * task_data->inputs_count[3]));
    size_ = task_data->inputs_count[0];
  }
  boost::mpi::broadcast(world_, size_, 0);
  return true;
}

bool MultiplMatrixMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[1] &&
            IsPowerOfTwo(task_data->inputs_count[0]) && IsPowerOfTwo(task_data->inputs_count[1]) &&
            IsPowerOfTwo(task_data->inputs_count[2]) && IsPowerOfTwo(task_data->inputs_count[3]));
  }
  return true;
}

bool MultiplMatrixMpi::RunImpl() {
  result_ = StrassenMultiplyMpi(first_input_, second_input_, size_);

  return true;
}

bool MultiplMatrixMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (unsigned int i = 0; i < task_data->outputs_count[0]; i++) {
      reinterpret_cast<double*>(task_data->outputs[0])[i] = result_[i];
    }
  }

  return true;
}

}  // namespace agafeev_s_strassen_alg_mpi
