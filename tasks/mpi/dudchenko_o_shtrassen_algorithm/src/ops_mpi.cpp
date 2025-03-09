#include "mpi/dudchenko_o_shtrassen_algorithm/include/ops_mpi.hpp"

#include <algorithm>
#include <array>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmSequential::PreProcessingImpl() {
  auto* inputs_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* inputs_b = reinterpret_cast<double*>(task_data->inputs[1]);
  size_ = static_cast<size_t>(std::sqrt(task_data->inputs_count[0]));
  matrixA_.assign(inputs_a, inputs_a + (size_ * size_));
  matrixB_.assign(inputs_b, inputs_b + (size_ * size_));
  result_.resize(size_ * size_);

  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmSequential::ValidationImpl() {
  return !task_data->inputs.empty() && task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->inputs_count[0] == static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) *
                                           static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmSequential::RunImpl() {
  result_ = StrassenSeq(matrixA_, matrixB_, size_);
  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmSequential::PostProcessingImpl() {
  auto* outputs = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(result_, outputs);
  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto* inputs_a = reinterpret_cast<double*>(task_data->inputs[0]);
    auto* inputs_b = reinterpret_cast<double*>(task_data->inputs[1]);
    size_ = static_cast<size_t>(std::sqrt(task_data->inputs_count[0]));
    matrixA_.assign(inputs_a, inputs_a + (size_ * size_));
    matrixB_.assign(inputs_b, inputs_b + (size_ * size_));
    result_.resize(size_ * size_);
  }
  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    return !task_data->inputs.empty() && task_data->inputs_count[0] == task_data->inputs_count[1] &&
           task_data->inputs_count[0] == static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) *
                                             static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) &&
           task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel::RunImpl() {
  result_ = StrassenMpi({.a = matrixA_, .b = matrixB_}, size_);
  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* outputs = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(result_, outputs);
  }
  return true;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_mpi::Add(const std::vector<double>& a, const std::vector<double>& b,
                                                             size_t n) {
  std::vector<double> result(n * n);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  return result;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_mpi::Subtract(const std::vector<double>& a,
                                                                  const std::vector<double>& b, size_t n) {
  std::vector<double> result(n * n);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
  return result;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_mpi::StrassenSeq(const std::vector<double>& a,
                                                                     const std::vector<double>& b, size_t n) {
  if (n == 1) {
    return {a[0] * b[0]};
  }

  size_t new_size = 1;
  while (new_size < n) {
    new_size *= 2;
  }

  std::vector<double> a_ext(new_size * new_size, 0.0);
  std::vector<double> b_ext(new_size * new_size, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      a_ext[(i * new_size) + j] = a[(i * n) + j];
      b_ext[(i * new_size) + j] = b[(i * n) + j];
    }
  }

  size_t half = new_size / 2;
  size_t half_squared = half * half;
  auto get_submatrix = [&](const std::vector<double>& m, size_t row, size_t col) {
    std::vector<double> sub(half_squared);
    for (size_t i = 0; i < half; ++i) {
      for (size_t j = 0; j < half; ++j) {
        sub[(i * half) + j] = m[((i + row) * new_size) + j + col];
      }
    }
    return sub;
  };

  auto a11 = get_submatrix(a_ext, 0, 0);
  auto a12 = get_submatrix(a_ext, 0, half);
  auto a21 = get_submatrix(a_ext, half, 0);
  auto a22 = get_submatrix(a_ext, half, half);
  auto b11 = get_submatrix(b_ext, 0, 0);
  auto b12 = get_submatrix(b_ext, 0, half);
  auto b21 = get_submatrix(b_ext, half, 0);
  auto b22 = get_submatrix(b_ext, half, half);

  auto m1 = StrassenSeq(Add(a11, a22, half), Add(b11, b22, half), half);
  auto m2 = StrassenSeq(Add(a21, a22, half), b11, half);
  auto m3 = StrassenSeq(a11, Subtract(b12, b22, half), half);
  auto m4 = StrassenSeq(a22, Subtract(b21, b11, half), half);
  auto m5 = StrassenSeq(Add(a11, a12, half), b22, half);
  auto m6 = StrassenSeq(Subtract(a21, a11, half), Add(b11, b12, half), half);
  auto m7 = StrassenSeq(Subtract(a12, a22, half), Add(b21, b22, half), half);

  std::vector<double> result_ext(new_size * new_size, 0.0);
  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      size_t idx = (i * half) + j;
      size_t ext_idx = (i * new_size) + j;
      result_ext[ext_idx] = m1[idx] + m4[idx] - m5[idx] + m7[idx];
      result_ext[ext_idx + half] = m3[idx] + m5[idx];
      result_ext[((i + half) * new_size) + j] = m2[idx] + m4[idx];
      result_ext[((i + half) * new_size) + j + half] = m1[idx] + m3[idx] - m2[idx] + m6[idx];
    }
  }

  std::vector<double> result(n * n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      result[(i * n) + j] = result_ext[(i * new_size) + j];
    }
  }

  return result;
}

namespace {
struct Size {
  size_t new_size;
  size_t half;
  size_t n;
};

size_t CalculateNewSize(size_t n) {
  size_t new_size = 1;
  while (new_size < n) {
    new_size *= 2;
  }
  return new_size;
}

std::vector<double> ConstructFinalResult(const std::vector<double>& m_global, Size size) {
  size_t half_squared = size.half * size.half;
  std::vector<double> result_ext(size.new_size * size.new_size, 0.0);
  for (size_t i = 0; i < size.half; ++i) {
    for (size_t j = 0; j < size.half; ++j) {
      size_t idx = (i * size.half) + j;
      result_ext[(i * size.new_size) + j] = m_global[idx] + m_global[(3 * half_squared) + idx] -
                                            m_global[(4 * half_squared) + idx] + m_global[(6 * half_squared) + idx];
      result_ext[(i * size.new_size) + j + size.half] =
          m_global[(2 * half_squared) + idx] + m_global[(4 * half_squared) + idx];
      result_ext[((i + size.half) * size.new_size) + j] =
          m_global[(1 * half_squared) + idx] + m_global[(3 * half_squared) + idx];
      result_ext[((i + size.half) * size.new_size) + j + size.half] =
          m_global[idx] - m_global[(1 * half_squared) + idx] + m_global[(2 * half_squared) + idx] +
          m_global[(5 * half_squared) + idx];
    }
  }

  std::vector<double> final_result(size.n * size.n);
  for (size_t i = 0; i < size.n; ++i) {
    for (size_t j = 0; j < size.n; ++j) {
      final_result[(i * size.n) + j] = result_ext[(i * size.new_size) + j];
    }
  }

  return final_result;
}
}  // namespace

std::vector<double> dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel::StrassenMpi(const Parametre& param,
                                                                                                size_t n) {
  if (world_.rank() > 6) {
    world_.split(1);
    return {};
  }

  boost::mpi::communicator active_comm = world_.split(0);
  int rank = active_comm.rank();
  int size = active_comm.size();

  boost::mpi::broadcast(active_comm, n, 0);

  size_t new_size = CalculateNewSize(n);

  std::vector<double> a_ext(new_size * new_size, 0.0);
  std::vector<double> b_ext(new_size * new_size, 0.0);
  if (rank == 0) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        a_ext[(i * new_size) + j] = param.a[(i * n) + j];
        b_ext[(i * new_size) + j] = param.b[(i * n) + j];
      }
    }
  }

  boost::mpi::broadcast(active_comm, a_ext.data(), (static_cast<int>(new_size * new_size)), 0);
  boost::mpi::broadcast(active_comm, b_ext.data(), (static_cast<int>(new_size * new_size)), 0);

  size_t half = new_size / 2;
  size_t half_squared = half * half;

  auto get_submatrix = [&](const std::vector<double>& m, size_t row, size_t col) {
    std::vector<double> sub(half_squared);
    for (size_t i = 0; i < half; ++i) {
      for (size_t j = 0; j < half; ++j) {
        sub[(i * half) + j] = m[((i + row) * new_size) + j + col];
      }
    }
    return sub;
  };

  auto a11 = get_submatrix(a_ext, 0, 0);
  auto a12 = get_submatrix(a_ext, 0, half);
  auto a21 = get_submatrix(a_ext, half, 0);
  auto a22 = get_submatrix(a_ext, half, half);
  auto b11 = get_submatrix(b_ext, 0, 0);
  auto b12 = get_submatrix(b_ext, 0, half);
  auto b21 = get_submatrix(b_ext, half, 0);
  auto b22 = get_submatrix(b_ext, half, half);

  std::vector<std::vector<double>> m(7, std::vector<double>(half_squared, 0.0));

  std::array<std::function<std::vector<double>()>, 7> computations = {
      [&]() { return StrassenSeq(Add(a11, a22, half), Add(b11, b22, half), half); },
      [&]() { return StrassenSeq(Add(a21, a22, half), b11, half); },
      [&]() { return StrassenSeq(a11, Subtract(b12, b22, half), half); },
      [&]() { return StrassenSeq(a22, Subtract(b21, b11, half), half); },
      [&]() { return StrassenSeq(Add(a11, a12, half), b22, half); },
      [&]() { return StrassenSeq(Subtract(a21, a11, half), Add(b11, b12, half), half); },
      [&]() { return StrassenSeq(Subtract(a12, a22, half), Add(b21, b22, half), half); }};

  for (int task = rank; task < 7; task += size) {
    m[task] = computations[task]();
  }

  std::vector<double> m_global(7 * half_squared, 0.0);
  for (size_t i = 0; i < 7; ++i) {
    boost::mpi::reduce(active_comm, m[i].data(), static_cast<int>(half_squared), m_global.data() + (i * half_squared),
                       std::plus(), 0);
  }

  if (rank == 0) {
    return ConstructFinalResult(m_global, {.new_size = new_size, .half = half, .n = n});
  }
  return {};
}