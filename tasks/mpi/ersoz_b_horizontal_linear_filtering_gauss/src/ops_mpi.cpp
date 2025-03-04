#include "mpi/ersoz_b_horizontal_linear_filtering_gauss/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <vector>

namespace {
inline double gaussianFunction(int i, int j, double sigma) {
  return 1 / (2 * M_PI * sigma * sigma) * exp(-(i * i + j * j) / (2 * sigma * sigma));
}

std::vector<std::vector<char>> gaussianFilter(const std::vector<std::vector<char>>& image, double sigma) {
  int Y = static_cast<int>(image.size());
  int X = static_cast<int>(image[0].size());
  int lineBlocks = Y - 2;
  int procs, rank, rem, lineBlocksPerProc;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  rem = lineBlocks % procs;
  lineBlocksPerProc = lineBlocks / procs;
  int* displs = new int[procs];
  int* scounts = new int[procs];
  int offset = 1;
  for (int i = 0; i < procs; i++) {
    displs[i] = offset;
    if (i < rem) {
      offset += lineBlocksPerProc + 1;
      scounts[i] = lineBlocksPerProc + 1;
    } else {
      offset += lineBlocksPerProc;
      scounts[i] = lineBlocksPerProc;
    }
  }
  char* pixels = new char[scounts[rank] * (X - 2)];
  for (int y = displs[rank]; y < displs[rank] + scounts[rank]; y++) {
    for (int x = 1; x < X - 1; x++) {
      double brightness = 0;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          brightness += gaussianFunction(i, j, sigma) * static_cast<int>(image[y + i][x + j]);
        }
      }
      pixels[(y - displs[rank]) * (X - 2) + (x - 1)] = static_cast<char>(brightness);
    }
  }
  std::vector<std::vector<char>> res;
  if (rank != 0) {
    MPI_Send(pixels, scounts[rank] * (X - 2), MPI_CHAR, 0, rank, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    for (int i = 0; i < procs; i++) {
      char* _p = nullptr;
      if (i != 0) {
        _p = new char[scounts[i] * (X - 2)];
        MPI_Recv(_p, scounts[i] * (X - 2), MPI_CHAR, i, i, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      }
      for (int y = displs[i]; y < displs[i] + scounts[i]; y++) {
        std::vector<char> line(X - 2);
        for (int x = 1; x < X - 1; x++) {
          if (i != 0) {
            line[x - 1] = _p[(y - displs[i]) * (X - 2) + (x - 1)];
          } else {
            line[x - 1] = pixels[(y - displs[i]) * (X - 2) + (x - 1)];
          }
        }
        res.push_back(line);
      }
      if (i != 0) {
        delete[] _p;
      }
    }
  }
  delete[] pixels;
  delete[] displs;
  delete[] scounts;
  return res;
}
}  // namespace

bool ersoz_b_test_task_mpi::TestTaskMPI::PreProcessingImpl() {
  // Reconstruct the 2D image from the flat input buffer.
  unsigned int input_size = task_data->inputs_count[0];
  img_size_ = static_cast<int>(std::sqrt(input_size));
  uint8_t* in_ptr = task_data->inputs[0];
  std::vector<char> flat(in_ptr, in_ptr + input_size);
  input_image_.resize(img_size_);
  for (int i = 0; i < img_size_; i++) {
    input_image_[i] = std::vector<char>(flat.begin() + i * img_size_, flat.begin() + (i + 1) * img_size_);
  }
  output_image_.resize(img_size_ - 2);
  for (int i = 0; i < img_size_ - 2; i++) {
    output_image_[i].resize(img_size_ - 2, 0);
  }
  return true;
}

bool ersoz_b_test_task_mpi::TestTaskMPI::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (static_cast<unsigned int>(img_size_ * img_size_) != input_size) return false;

  if (task_data->outputs_count[0] != static_cast<unsigned int>((img_size_ - 2) * (img_size_ - 2))) return false;
  return true;
}

bool ersoz_b_test_task_mpi::TestTaskMPI::RunImpl() {
  output_image_ = gaussianFilter(input_image_, sigma_);
  world_.barrier();
  return true;
}

bool ersoz_b_test_task_mpi::TestTaskMPI::PostProcessingImpl() {
  uint8_t* out_ptr = task_data->outputs[0];
  int index = 0;
  for (const auto& row : output_image_) {
    for (char pixel : row) {
      out_ptr[index++] = static_cast<uint8_t>(pixel);
    }
  }
  return true;
}
