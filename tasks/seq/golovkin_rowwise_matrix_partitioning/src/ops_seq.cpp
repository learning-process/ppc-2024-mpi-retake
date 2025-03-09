// Golovkin Maksim Task#2

#include "seq/golovkin_rowwise_matrix_partitioning/include/ops_seq.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

using namespace golovkin_rowwise_matrix_partitioning;

MatrixMultiplicationTask::MatrixMultiplicationTask(const std::shared_ptr<ppc::core::TaskData>& taskData)
    : ppc::core::Task(taskData), taskData_(taskData) {}

bool MatrixMultiplicationTask::validation() {
  internal_order_test();
  if (!taskData_ || taskData_->inputs.size() < 2 || taskData_->outputs.empty()) {
    return false;
  }

  auto* matrixA = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[0]);
  auto* matrixB = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[1]);
  if (matrixA == nullptr || matrixB == nullptr || matrixA->empty() || matrixB->empty()) {
    return false;
  }

  if (matrixA->at(0).size() != matrixB->size()) {
    return false;
  }

  return true;
}

bool MatrixMultiplicationTask::pre_processing() {
  internal_order_test();
  auto* matrixA = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[0]);
  auto* matrixB = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[1]);
  auto* result = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->outputs[0]);

  if (matrixA == nullptr || matrixB == nullptr || result == nullptr) {
    return false;
  }

  size_t rowsA = matrixA->size();
  size_t colsB = matrixB->at(0).size();
  result->resize(rowsA, std::vector<double>(colsB, 0.0));
  return true;
}

static const size_t BLOCK_SIZE = 32;

static bool multiplier(std::vector<std::vector<double>>& matrixA, std::vector<std::vector<double>>& matrixB,
                       std::vector<std::vector<double>>& result) {
  size_t rowsA = matrixA.size();
  size_t colsA = matrixA[0].size();
  size_t colsB = matrixB[0].size();

  for (size_t iBlock = 0; iBlock < rowsA; iBlock += BLOCK_SIZE) {
    for (size_t kBlock = 0; kBlock < colsA; kBlock += BLOCK_SIZE) {
      for (size_t jBlock = 0; jBlock < colsB; jBlock += BLOCK_SIZE) {
        for (size_t i = iBlock; i < std::min(iBlock + BLOCK_SIZE, rowsA); ++i) {
          auto& rowRes = result[i];
          for (size_t k = kBlock; k < std::min(kBlock + BLOCK_SIZE, colsA); ++k) {
            double a_ik = matrixA[i][k];
            const auto& rowB = matrixB[k];
            for (size_t j = jBlock; j < std::min(jBlock + BLOCK_SIZE, colsB); ++j) {
              rowRes[j] += a_ik * rowB[j];
            }
          }
        }
      }
    }
  }
  return true;
}

bool MatrixMultiplicationTask::run() {
  internal_order_test();
  auto* matrixA = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[0]);
  auto* matrixB = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[1]);
  auto* result = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->outputs[0]);

  if (matrixA == nullptr || matrixB == nullptr || result == nullptr) {
    return false;
  }

  return multiplier(*matrixA, *matrixB, *result);
}

bool MatrixMultiplicationTask::post_processing() {
  internal_order_test();
  if (!result_.empty()) {
    const double epsilon = 1e-9;

    for (size_t i = 0; i < result_.size(); ++i) {
      for (size_t j = 0; j < result_[0].size(); ++j) {
        if (std::abs(result_[i][j]) < epsilon) {
          result_[i][j] = 0.0;
        }
      }
    }

    return true;
  }
  return false;
}
