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
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool is_root = (rank == 0);

  if (is_root && (task_data->inputs.empty() || task_data->inputs_count.empty())) {
    std::cerr << "[ERROR] Root process has empty inputs or inputs_count.\n";
    return false;
  }

  input_size_ = is_root ? static_cast<int>(task_data->inputs_count[0]) : 0;
  int rows = is_root ? static_cast<int>(task_data->inputs_count[0] / task_data->inputs_count[1]) : 0;
  int cols = is_root ? static_cast<int>(task_data->inputs_count[1]) : 0;

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0 || cols == 0) {
    std::cerr << "[ERROR] Rows or columns size is zero.\n";
    return false;
  }

  rc_size_ = rows;

  if (is_root) {
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size_);
  } else {
    input_.resize(rows * cols);
  }

  std::cout << "[INFO] Process " << rank << ": Broadcasting input data...\n";
  int bcast_status = MPI_Bcast(input_.data(), rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
  if (bcast_status != MPI_SUCCESS) {
    std::cerr << "[ERROR] MPI_Bcast failed in PreProcessingImpl. Status: " << bcast_status << "\n";
    return false;
  }
  std::cout << "[INFO] Process " << rank << ": MPI_Bcast completed successfully.\n";

  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  unsigned int input_count = 0;
  unsigned int output_count = 0;

  if (!task_data->inputs_count.empty()) {
    input_count = task_data->inputs_count[0];
  }
  if (!task_data->outputs_count.empty()) {
    output_count = task_data->outputs_count[0];
  }

  // Broadcasting input_count and output_count
  MPI_Bcast(&input_count, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&output_count, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  std::cout << "[INFO] Process " << rank << ": Broadcasting validation data (input_count = " << input_count
            << ", output_count = " << output_count << ")\n";

  bool local_valid = (input_count > 0) && (output_count > 0) && (input_count == output_count);

  // Broadcasting local_valid status
  MPI_Bcast(&local_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "[INFO] Process 0: Validation result broadcasted: " << local_valid << "\n";
  } else {
    std::cout << "[INFO] Process " << rank << ": Received validation result: " << local_valid << "\n";
  }

  return local_valid;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::RunImpl() {
  if (rc_size_ == 0) {
    std::cerr << "[ERROR] rc_size_ is zero. Exiting.\n";
    return false;
  }

  int rows = rc_size_;
  int cols = input_size_ / rows;
  int min_label = 2;
  std::unordered_map<int, int> label_parent;

  int num_procs = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows_per_proc = rows / num_procs;
  int remainder = rows % num_procs;
  int start_row = (rank < remainder) ? rank * (rows_per_proc + 1) : (rank * rows_per_proc) + remainder;
  int end_row = start_row + ((rank < remainder) ? (rows_per_proc + 1) : rows_per_proc);

  if (end_row <= start_row) {
    std::cerr << "[ERROR] Invalid row range for process " << rank << ": start_row = " << start_row
              << ", end_row = " << end_row << "\n";
    return false;
  }

  std::vector<int> labeled_image(rows * cols, 0);
  Labeling(input_, labeled_image, rows, cols, min_label, label_parent, start_row, end_row);

  std::cout << "[INFO] Process " << rank << ": Labeling completed, syncing...\n";
  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::PostProcessingImpl() {
  if (!task_data->outputs.empty() && task_data->outputs[0] != nullptr) {
    std::ranges::copy(output_.begin(), output_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
    return true;
  }
  std::cerr << "[ERROR] Output is null or empty!\n";
  return false;
}