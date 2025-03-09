#include "seq/agafeev_s_strassen_alg/include/strassen_seq.hpp"
#include <cassert>
#include<iostream>

static bool isPowerOfTwo(int n){
    return n && !(n & (n - 1));
}

namespace agafeev_s_strassen_alg_seq {

// std::vector<double> padMatrix(const std::vector<double>& matrix, int currentSize, int targetSize) {
//     std::vector<double> padded(targetSize * targetSize, 0.0);
//     for (int i = 0; i < currentSize; ++i) {
//         for (int j = 0; j < currentSize; ++j) {
//             padded[i * targetSize + j] = matrix[i * currentSize + j];
//         }
//     }
//     return padded;
// }

std::vector<double> addMatrices(const std::vector<double>& A, const std::vector<double>& B, int n) {
    // assert(A.size() == n * n && B.size() == n * n);
    std::vector<double> C(n * n);
    for (int i = 0; i < n * n; ++i) {
        C[i] = A[i] + B[i];
    }
    return C;
}

std::vector<double> subtractMatrices(const std::vector<double>& A, const std::vector<double>& B, int n) {
    // assert(A.size() == n * n && B.size() == n * n);
    std::vector<double> C(n * n);
    for (int i = 0; i < n * n; ++i) {
        C[i] = A[i] - B[i];
    }
    return C;
}

void splitMatrix(const std::vector<double>& A, std::vector<double>& A11, std::vector<double>& A12, std::vector<double>& A21, std::vector<double>& A22, int n) {
    // assert(A.size() == n * n);
    int half = n / 2;
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            A11[i * half + j] = A[i * n + j];
            A12[i * half + j] = A[i * n + (j + half)];
            A21[i * half + j] = A[(i + half) * n + j];
            A22[i * half + j] = A[(i + half) * n + (j + half)];
        }
    }
}

std::vector<double> mergeMatrices(const std::vector<double>& A11, const std::vector<double>& A12, const std::vector<double>& A21, const std::vector<double>& A22, int n) {
    int half = n / 2;
    std::vector<double> A(n * n);
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            A[i * n + j] = A11[i * half + j];
            A[i * n + (j + half)] = A12[i * half + j];
            A[(i + half) * n + j] = A21[i * half + j];
            A[(i + half) * n + (j + half)] = A22[i * half + j];
        }
    }
    return A;
}

std::vector<double> strassenMultiply(const std::vector<double>& A, const std::vector<double>& B, int n) {
    // std::cout<<"A size: "<<A.size()<<" B size: "<<B.size()<<" n: "<<n<<std::endl;
    // assert(A.size() == n * n && B.size() == n * n);
    if (n == 1) {
        return {A[0] * B[0]};
    }

    int half = n / 2;

    std::vector<double> A11(half * half), A12(half * half), A21(half * half), A22(half * half);
    std::vector<double> B11(half * half), B12(half * half), B21(half * half), B22(half * half);
    splitMatrix(A, A11, A12, A21, A22, n);
    splitMatrix(B, B11, B12, B21, B22, n);

    auto P1 = strassenMultiply(addMatrices(A11, A22, half), addMatrices(B11, B22, half), half);
    auto P2 = strassenMultiply(addMatrices(A21, A22, half), B11, half);
    auto P3 = strassenMultiply(A11, subtractMatrices(B12, B22, half), half);
    auto P4 = strassenMultiply(A22, subtractMatrices(B21, B11, half), half);
    auto P5 = strassenMultiply(addMatrices(A11, A12, half), B22, half);
    auto P6 = strassenMultiply(subtractMatrices(A21, A11, half), addMatrices(B11, B12, half), half);
    auto P7 = strassenMultiply(subtractMatrices(A12, A22, half), addMatrices(B21, B22, half), half);

    auto C11 = addMatrices(subtractMatrices(addMatrices(P1, P4, half), P5, half), P7, half);
    auto C12 = addMatrices(P3, P5, half);
    auto C21 = addMatrices(P2, P4, half);
    auto C22 = addMatrices(subtractMatrices(addMatrices(P1, P3, half), P2, half), P6, half);

    return mergeMatrices(C11, C12, C21, C22, n);
}

bool MultiplMatrixSequental::PreProcessingImpl() {
  first_input_.clear();
  second_input_.clear();
  auto* temp_ptr1 = reinterpret_cast<double*>(task_data->inputs[0]);
  first_input_.insert(first_input_.begin(), temp_ptr1, temp_ptr1 + task_data->inputs_count[0]*task_data->inputs_count[1]);
  auto* temp_ptr2 = reinterpret_cast<double*>(task_data->inputs[1]);
  second_input_.insert(second_input_.begin(), temp_ptr2, temp_ptr2 + task_data->inputs_count[2]*task_data->inputs_count[3]);
  size_ = task_data->inputs_count[0];

  return true;
}

bool MultiplMatrixSequental::ValidationImpl() {
  return (task_data->outputs_count[0] == task_data->inputs_count[0]*task_data->inputs_count[1] && isPowerOfTwo(task_data->inputs_count[0]) && isPowerOfTwo(task_data->inputs_count[1]) && isPowerOfTwo(task_data->inputs_count[2]) && isPowerOfTwo(task_data->inputs_count[3]));
}

bool MultiplMatrixSequental::RunImpl() {
  result_ = strassenMultiply(first_input_, second_input_, size_);

  return true;
}

bool MultiplMatrixSequental::PostProcessingImpl() {
  for (unsigned int i = 0; i < task_data->outputs_count[0]; i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = result_[i];
  }
  return true;
}


}  // namespace agafeev_s_strassen_alg_seq
