#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>

// Function to get the root of a label with path compression
int karaseva_e_binaryimage_mpi::TestTaskMPI::GetRootLabel(std::unordered_map<int, int>& label_parent, int label) {
  if (!label_parent.contains(label)) {
    label_parent[label] = label;
  } else if (label_parent[label] != label) {
    label_parent[label] = GetRootLabel(label_parent, label_parent[label]);  // Path compression
  }
  return label_parent[label];
}

// Function to union two labels
void karaseva_e_binaryimage_mpi::TestTaskMPI::UnionLabels(std::unordered_map<int, int>& label_parent, int label1,
                                                          int label2) {
  int root1 = GetRootLabel(label_parent, label1);
  int root2 = GetRootLabel(label_parent, label2);
  if (root1 != root2) {
    label_parent[root2] = root1;  // Union
  }
}

// Function to process neighbors of a pixel and add them to a list
void karaseva_e_binaryimage_mpi::TestTaskMPI::ProcessNeighbors(int x, int y, int rows, int cols,
                                                               const std::vector<int>& labeled_image,
                                                               std::vector<int>& neighbors) {
  int dx[] = {-1, 0, -1};
  int dy[] = {0, -1, 1};

  for (int i = 0; i < 3; ++i) {
    int nx = x + dx[i];
    int ny = y + dy[i];
    if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && labeled_image[(nx * cols) + ny] >= 2) {
      neighbors.push_back(labeled_image[(nx * cols) + ny]);
    }
  }
}

// Function to assign label to a pixel and perform union of labels
void karaseva_e_binaryimage_mpi::TestTaskMPI::AssignLabelToPixel(int pos, std::vector<int>& labeled_image,
                                                                 std::unordered_map<int, int>& label_parent,
                                                                 int& label_counter,
                                                                 const std::vector<int>& neighbors) {
  if (neighbors.empty()) {
    labeled_image[pos] = label_counter++;
  } else {
    int min_neighbor = *std::ranges::min_element(neighbors);
    labeled_image[pos] = min_neighbor;
    for (int n : neighbors) {
      UnionLabels(label_parent, min_neighbor, n);
    }
  }
}

// Main labeling function
void karaseva_e_binaryimage_mpi::TestTaskMPI::Labeling(std::vector<int>& image, std::vector<int>& labeled_image,
                                                       int rows, int cols, int min_label,
                                                       std::unordered_map<int, int>& label_parent, int start_row,
                                                       int end_row) {
  int label_counter = min_label;

  for (int x = start_row; x < end_row; ++x) {
    for (int y = 0; y < cols; ++y) {
      int pos = (x * cols) + y;
      if (image[pos] == 0 || labeled_image[pos] >= 2) {
        std::vector<int> neighbors;
        ProcessNeighbors(x, y, rows, cols, labeled_image, neighbors);
        AssignLabelToPixel(pos, labeled_image, label_parent, label_counter, neighbors);
      }
    }
  }

  // Local root search after labeling
  for (int x = start_row; x < end_row; ++x) {
    for (int y = 0; y < cols; ++y) {
      int pos = (x * cols) + y;
      if (labeled_image[pos] >= 2) {
        labeled_image[pos] = GetRootLabel(label_parent, labeled_image[pos]);
      }
    }
  }
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->inputs_count.empty()) {
    return false;
  }

  auto input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  if (in_ptr == nullptr) {
    return false;
  }

  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  if (task_data->outputs_count.empty()) {
    task_data->outputs_count.push_back(input_size);
  }

  auto output_size = static_cast<unsigned int>(task_data->outputs_count[0]);
  output_ = std::vector<int>(output_size, 0);

  rc_size_ = static_cast<int>(std::sqrt(input_size));

  MPI_Bcast(&rc_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  input_.resize(rc_size_ * rc_size_);
  MPI_Bcast(input_.data(), static_cast<int>(input_.size()), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::ValidationImpl() {
  int rank = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "Rank: " << rank << " inputs.size(): " << task_data->inputs.size() << '\n';
  std::cout << "Rank: " << rank << " outputs.size(): " << task_data->outputs.size() << '\n';

  if (!task_data->inputs.empty()) {
    std::cout << "Rank: " << rank << " inputs_count[0]: " << task_data->inputs_count[0] << '\n';
  }
  if (!task_data->outputs.empty()) {
    std::cout << "Rank: " << rank << " outputs_count[0]: " << task_data->outputs_count[0] << '\n';
  }

  int input_count = task_data->inputs_count[0];
  int output_count = task_data->outputs_count[0];

  bool valid = !task_data->inputs.empty() && !task_data->outputs.empty() && input_count == output_count;

  MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return valid;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::RunImpl() {
  int rows = rc_size_;
  int cols = rc_size_;
  int min_label = 2;  // Starting label
  std::unordered_map<int, int> label_parent;

  int num_procs = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows_per_proc = rows / num_procs;
  int remainder = rows % num_procs;
  int start_row = (rank < remainder) ? rank * (rows_per_proc + 1) : (rank * rows_per_proc) + remainder;
  int end_row = start_row + ((rank < remainder) ? (rows_per_proc + 1) : rows_per_proc);

  std::vector<int> labeled_image(rows * cols, 0);

  Labeling(input_, labeled_image, rows, cols, min_label, label_parent, start_row, end_row);

  std::vector<int> recv_counts(num_procs);
  std::vector<int> displs(num_procs);

  for (int i = 0; i < num_procs; ++i) {
    recv_counts[i] = (i < remainder) ? (rows_per_proc + 1) * cols : rows_per_proc * cols;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + recv_counts[i - 1];
  }

  MPI_Gatherv(labeled_image.data() + (start_row * cols), recv_counts[rank], MPI_INT, labeled_image.data(),
              recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::PostProcessingImpl() {
  if (!task_data->outputs.empty() && task_data->outputs[0] != nullptr) {
    for (size_t i = 0; i < output_.size(); ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
    }
    return true;
  }
  return false;
}