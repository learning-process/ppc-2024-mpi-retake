#define OMPI_SKIP_MPICXX

#include "mpi/ersoz_b_horizontal_linear_filtering_gauss/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

inline double GaussianFunction(int i, int j, double sigma) {
  return 1.0 / (2.0 * M_PI * sigma * sigma) * exp(-((i * i) + (j * j)) / (2.0 * sigma * sigma));
}

inline char ComputePixel(const std::vector<std::vector<char>>& image, int y, int x, double sigma) {
  double brightness = 0.0;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      brightness += GaussianFunction(i, j, sigma) * static_cast<int>(image[y + i][x + j]);
    }
  }
  return static_cast<char>(brightness);
}

void ComputeDisplsAndScounts(int line_blocks, int procs, int& rem, int& line_blocks_per_proc, std::vector<int>& displs,
                             std::vector<int>& scounts) {
  rem = line_blocks % procs;
  line_blocks_per_proc = line_blocks / procs;
  int offset = 1;
  displs.resize(procs);
  scounts.resize(procs);
  for (int i = 0; i < procs; i++) {
    displs[i] = offset;
    if (i < rem) {
      offset += line_blocks_per_proc + 1;
      scounts[i] = line_blocks_per_proc + 1;
    } else {
      offset += line_blocks_per_proc;
      scounts[i] = line_blocks_per_proc;
    }
  }
}

std::vector<std::vector<char>> GaussianFilter(const std::vector<std::vector<char>>& image, double sigma) {
  int y_dim = static_cast<int>(image.size());
  int x_dim = static_cast<int>(image[0].size());
  int line_blocks = y_dim - 2;
  int procs = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int rem = 0;
  int line_blocks_per_proc = 0;
  std::vector<int> displs;
  std::vector<int> scounts;
  ComputeDisplsAndScounts(line_blocks, procs, rem, line_blocks_per_proc, displs, scounts);

  int local_size = scounts[rank] * (x_dim - 2);
  std::vector<char> pixels(local_size, 0);

  for (int y = displs[rank]; y < displs[rank] + scounts[rank]; y++) {
    for (int x = 1; x < x_dim - 1; x++) {
      pixels[((y - displs[rank]) * (x_dim - 2)) + (x - 1)] = ComputePixel(image, y, x, sigma);
    }
  }

  std::vector<std::vector<char>> res;
  if (rank != 0) {
    MPI_Send(pixels.data(), local_size, MPI_CHAR, 0, rank, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    for (int i = 0; i < procs; i++) {
      std::vector<char> temp;
      if (i != 0) {
        int temp_size = scounts[i] * (x_dim - 2);
        temp.resize(temp_size);
        MPI_Recv(temp.data(), temp_size, MPI_CHAR, i, i, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      }
      for (int y = displs[i]; y < displs[i] + scounts[i]; y++) {
        std::vector<char> line(x_dim - 2, 0);
        for (int x = 1; x < x_dim - 1; x++) {
          if (i != 0) {
            line[x - 1] = temp[((y - displs[i]) * (x_dim - 2)) + (x - 1)];
          } else {
            line[x - 1] = pixels[((y - displs[i]) * (x_dim - 2)) + (x - 1)];
          }
        }
        res.emplace_back(std::move(line));
      }
    }
  }
  return res;
}

}  // namespace

bool ersoz_b_test_task_mpi::TestTaskMPI::PreProcessingImpl() {
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
  int computed_size = static_cast<int>(std::sqrt(input_size));
  if (static_cast<unsigned int>(computed_size * computed_size) != input_size) {
    return false;
  }
  if (task_data->outputs_count[0] != static_cast<unsigned int>((computed_size - 2) * (computed_size - 2))) {
    return false;
  }
  img_size_ = computed_size;
  return true;
}

bool ersoz_b_test_task_mpi::TestTaskMPI::RunImpl() {
  output_image_ = GaussianFilter(input_image_, sigma_);
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
