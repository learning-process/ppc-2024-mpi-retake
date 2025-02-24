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
    labeled_image[pos] = label_counter++;  // No neighbors, assign a new label
  } else {
    int min_neighbor = *std::ranges::min_element(neighbors);  // Find the smallest neighbor label
    labeled_image[pos] = min_neighbor;
    for (int n : neighbors) {
      UnionLabels(label_parent, min_neighbor, n);  // Union current label with neighbors
    }
  }
}

// Main labeling function
void karaseva_e_binaryimage_mpi::TestTaskMPI::Labeling(std::vector<int>& image, std::vector<int>& labeled_image,
                                                       int rows, int cols, int min_label,
                                                       std::unordered_map<int, int>& label_parent, int start_row,
                                                       int end_row) {
  int label_counter = min_label;

  // Iterate through the image rows assigned to the current process
  for (int x = start_row; x < end_row; ++x) {
    for (int y = 0; y < cols; ++y) {
      int pos = (x * cols) + y;
      if (image[pos] == 0 || labeled_image[pos] >= 2) {  // Skip background or already labeled pixels
        std::vector<int> neighbors;
        ProcessNeighbors(x, y, rows, cols, labeled_image, neighbors);                    // Find neighbors of the pixel
        AssignLabelToPixel(pos, labeled_image, label_parent, label_counter, neighbors);  // Label the pixel
      }
    }
  }

  // Local root search after labeling
  for (int x = start_row; x < end_row; ++x) {
    for (int y = 0; y < cols; ++y) {
      int pos = (x * cols) + y;
      if (labeled_image[pos] >= 2) {
        labeled_image[pos] = GetRootLabel(label_parent, labeled_image[pos]);  // Apply path compression for final labels
      }
    }
  }
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->inputs_count.empty()) {
    return false;
  }

  // Image size is the number of elements
  input_size_ = static_cast<int>(task_data->inputs_count[0]);  // Initialize input_size_

  // Image dimensions (assuming they are passed through inputs_count)
  int rows = static_cast<int>(task_data->inputs_count[0] / task_data->inputs_count[1]);  // Number of rows
  int cols = static_cast<int>(task_data->inputs_count[1]);                               // Number of columns

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  if (in_ptr == nullptr) {
    return false;
  }

  input_ = std::vector<int>(in_ptr, in_ptr + input_size_);

  if (task_data->outputs_count.empty()) {
    task_data->outputs_count.push_back(input_size_);
  }

  rc_size_ = rows;

  // Synchronize all processes before data transmission
  MPI_Barrier(MPI_COMM_WORLD);

  // Broadcast image dimensions to all processes
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Broadcast the image data
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

  unsigned int input_count = task_data->inputs_count[0];
  unsigned int output_count = task_data->outputs_count[0];

  bool valid = !task_data->inputs.empty() && !task_data->outputs.empty() && input_count == output_count;

  MPI_Barrier(MPI_COMM_WORLD);

  // Check data validity
  int result = MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  if (result != MPI_SUCCESS) {
    std::cerr << "MPI_Bcast failed with error code: " << result << '\n';
    return false;
  }

  return valid;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::RunImpl() {
  int rows = rc_size_;
  int cols = input_size_ / rows;
  int min_label = 2;
  std::unordered_map<int, int> label_parent;

  int num_procs = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Divide rows evenly between processes
  int rows_per_proc = rows / num_procs;
  int remainder = rows % num_procs;
  int start_row = (rank < remainder) ? rank * (rows_per_proc + 1) : (rank * rows_per_proc) + remainder;
  int end_row = start_row + ((rank < remainder) ? (rows_per_proc + 1) : rows_per_proc);

  std::vector<int> labeled_image(rows * cols, 0);

  // Labeling for assigned rows
  Labeling(input_, labeled_image, rows, cols, min_label, label_parent, start_row, end_row);

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