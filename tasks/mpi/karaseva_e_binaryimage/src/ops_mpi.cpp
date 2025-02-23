#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <vector>

// Function to get the root of a label with path compression
int karaseva_e_binaryimage_mpi::TestTaskMPI::GetRootLabel(std::unordered_map<int, int>& label_parent, int label) {
  if (label_parent.find(label) == label_parent.end()) {
    label_parent[label] = label;
    return label;
  }
  if (label_parent[label] != label) {
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
    int nx = x + dx[i], ny = y + dy[i];
    if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && labeled_image[nx * cols + ny] > 1) {
      neighbors.push_back(labeled_image[nx * cols + ny]);
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
    int min_neighbor = *std::min_element(neighbors.begin(), neighbors.end());
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
      int pos = x * cols + y;
      if (image[pos] == 0 || labeled_image[pos] > 1) {
        std::vector<int> neighbors;
        ProcessNeighbors(x, y, rows, cols, labeled_image, neighbors);
        AssignLabelToPixel(pos, labeled_image, label_parent, label_counter, neighbors);
      }
    }
  }

  // Local root search after labeling
  for (int x = start_row; x < end_row; ++x) {
    for (int y = 0; y < cols; ++y) {
      int pos = x * cols + y;
      if (labeled_image[pos] > 1) {
        labeled_image[pos] = GetRootLabel(label_parent, labeled_image[pos]);
      }
    }
  }
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  // Check that the outputs_count vector is not empty
  if (task_data->outputs_count.empty()) {
    return false;  // or handle the error
  }

  unsigned int output_size = static_cast<unsigned int>(task_data->outputs_count[0]);  // Type cast if necessary
  output_ = std::vector<int>(output_size, 0);  // The size of the output array matches the size of the input

  rc_size_ =
      static_cast<int>(std::sqrt(input_size));  // rc_size should be equal to sqrt(input_size), assuming a square image
  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::ValidationImpl() {
  // Ensure the input and output sizes match
  return task_data->inputs_count[0] == task_data->outputs_count[0];
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
  int start_row = rank * rows_per_proc;
  int end_row = (rank == num_procs - 1) ? rows : (rank + 1) * rows_per_proc;

  std::vector<int> labeled_image(rows * cols, 0);

  // Initial labeling for the local area
  Labeling(input_, labeled_image, rows, cols, min_label, label_parent, start_row, end_row);

  // Prepare "ghost" cells for boundary synchronization
  std::vector<int> ghost_cells_left(cols, 0);
  std::vector<int> ghost_cells_right(cols, 0);

  if (rank > 0) {  // Send left boundary to the previous rank
    MPI_Send(labeled_image.data() + start_row * cols, cols, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
  }

  if (rank < num_procs - 1) {  // Send right boundary to the next rank
    MPI_Send(labeled_image.data() + (end_row - 1) * cols, cols, MPI_INT, rank + 1, 1, MPI_COMM_WORLD);
  }

  if (rank != 0) {
    MPI_Recv(ghost_cells_left.data(), cols, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (rank != num_procs - 1) {
    MPI_Recv(ghost_cells_right.data(), cols, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Recalculate labels considering "ghost" cells (boundary pixels)
  for (int i = 0; i < cols; ++i) {
    if (start_row > 0 && labeled_image[start_row * cols + i] > 1) {
      UnionLabels(label_parent, labeled_image[start_row * cols + i], ghost_cells_left[i]);
    }
    if (end_row < rows && labeled_image[(end_row - 1) * cols + i] > 1) {
      UnionLabels(label_parent, labeled_image[(end_row - 1) * cols + i], ghost_cells_right[i]);
    }
  }

  Labeling(input_, labeled_image, rows, cols, min_label, label_parent, start_row, end_row);

  // Gather labels from all processes
  MPI_Allgather(labeled_image.data() + start_row * cols, rows_per_proc * cols, MPI_INT, labeled_image.data(),
                rows_per_proc * cols, MPI_INT, MPI_COMM_WORLD);

  // Perform final post-processing (if needed) and write the result
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}