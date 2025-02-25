#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
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
    labeled_image[pos] = label_counter++;  // No neighbors, assign a new label
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

  if (start_row >= end_row) {
    return;
  }

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
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool is_root = (rank == 0);

  if (is_root && (task_data->inputs.empty() || task_data->inputs_count.empty())) {
    std::cerr << "[Rank " << rank << "] [ERROR] Root process has empty inputs or inputs_count.\n";
    return false;
  }

  int rows = 0;
  int cols = 0;
  if (is_root) {
    rows = static_cast<int>(task_data->inputs_count[0]);
    cols = static_cast<int>(task_data->inputs_count[1]);
  }

  // Broadcasting the size and image dimensions to all processes
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::cerr << "[Rank " << rank << "] After Bcast: rows = " << rows << ", cols = " << cols << '\n';

  // Ensure valid image dimensions
  if (rows == 0 || cols == 0) {
    std::cerr << "[Rank " << rank << "] [ERROR] Invalid image dimensions: " << rows << "x" << cols
              << '\n';
        return false;
  }

  int input_size = rows * cols;

  // Broadcasting the image data
  if (is_root) {
    if (task_data->inputs[0] == nullptr) {
      std::cerr << "[Rank " << rank << "] [ERROR] Input data pointer is null.\n";
      return false;
    }
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  } else {
    input_.resize(input_size);
  }

  int result = MPI_Bcast(input_.data(), input_size, MPI_INT, 0, MPI_COMM_WORLD);
  if (result != MPI_SUCCESS) {
    std::cerr << "[Rank " << rank << "] Error broadcasting image data. MPI_Bcast failed.\n";
    return false;
  }

  std::cerr << "[Rank " << rank << "] Image data broadcasted successfully.\n";

  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] != 0 && task_data->outputs_count[0] != 0;
  }
  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = static_cast<int>(task_data->inputs_count[0]);
  int cols = static_cast<int>(task_data->inputs_count[1]);
  int num_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  int local_rows = rows / num_processes;
  if (local_rows == 0) {
    std::cerr << "[Rank " << rank << "] [ERROR] local_rows is zero. Check rows and num_processes.\n";
    return false;
  }

  int start_row = rank * local_rows;
  int end_row = (rank + 1) * local_rows;

  std::cerr << "[Rank " << rank << "] rows: " << rows << ", cols: " << cols << ", local_rows: " << local_rows
            << ", start_row: " << start_row << ", end_row: " << end_row << '\n';

  std::unordered_map<int, int> label_parent;
  local_labeled_image_.resize(local_rows * cols, 0);

  Labeling(input_, local_labeled_image_, rows, cols, 2, label_parent, start_row, end_row);

  std::vector<int> recv_counts(num_processes);
  std::vector<int> displs(num_processes);

  int local_size = local_rows * cols;
  MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    displs[0] = 0;
    for (int i = 1; i < num_processes; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
  }

  int result =
      MPI_Gatherv(local_labeled_image_.data(), local_size, MPI_INT, reinterpret_cast<int*>(task_data->outputs[0]),
                  recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (result != MPI_SUCCESS) {
    std::cerr << "[Rank " << rank << "] Error gathering labeled image data.\n";
    return false;
  }

  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(local_labeled_image_.begin(), local_labeled_image_.end(), output_ptr);
  }

  return true;
}