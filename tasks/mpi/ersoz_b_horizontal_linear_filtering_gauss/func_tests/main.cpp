#define OMPI_SKIP_MPICXX

#include <gtest/gtest.h>

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <mpi.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/ersoz_b_horizontal_linear_filtering_gauss/include/ops_mpi.hpp"

namespace {

std::vector<char> GenerateTestInput(int k_n) {
  std::vector<char> in(k_n * k_n, 0);
  for (int i = 0; i < k_n; i++) {
    for (int j = 0; j < k_n; j++) {
      // Parentheses clarify the order of operations.
      in[(i * k_n) + j] = static_cast<char>((i + j) % 256);
    }
  }
  return in;
}

// Convert the flat vector to a 2D image.
std::vector<std::vector<char>> ConvertToImage(const std::vector<char>& flat, int k_n) {
  std::vector<std::vector<char>> image;
  image.reserve(k_n);
  for (int i = 0; i < k_n; i++) {
    // Parentheses clarify the arithmetic.
    image.emplace_back(flat.begin() + (i * k_n), flat.begin() + ((i + 1) * k_n));
  }
  return image;
}

// Compute a sequential Gaussian filter on the image.
std::vector<std::vector<char>> ComputeSequentialFilter(const std::vector<std::vector<char>>& image, double sigma) {
  int y_dim = static_cast<int>(image.size());
  int x_dim = static_cast<int>(image[0].size());
  std::vector<std::vector<char>> res;
  res.reserve(y_dim - 2);
  for (int y = 1; y < y_dim - 1; y++) {
    std::vector<char> line;
    line.reserve(x_dim - 2);
    for (int x = 1; x < x_dim - 1; x++) {
      double brightness = 0.0;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          brightness += (1.0 / (2.0 * M_PI * sigma * sigma)) * exp(-((i * i) + (j * j)) / (2.0 * sigma * sigma)) *
                        static_cast<int>(image[y + i][x + j]);
        }
      }
      line.emplace_back(static_cast<char>(brightness));
    }
    res.push_back(std::move(line));
  }
  return res;
}

// Convert the flat output vector back to a 2D image.
std::vector<std::vector<char>> ConvertOutputToImage(const std::vector<char>& out, int k_n) {
  std::vector<std::vector<char>> result;
  result.reserve(k_n - 2);
  for (int i = 0; i < k_n - 2; i++) {
    result.emplace_back(out.begin() + (i * (k_n - 2)), out.begin() + ((i + 1) * (k_n - 2)));
  }
  return result;
}

}  // namespace

TEST(ersoz_b_test_task_mpi, test_gaussian_filter_small) {
  constexpr int kN = 16;
  auto in = GenerateTestInput(kN);
  auto image = ConvertToImage(in, kN);
  auto expected = ComputeSequentialFilter(image, 0.5);

  std::vector<char> out((kN - 2) * (kN - 2), 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.push_back(in.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.push_back(out.size());

  ersoz_b_test_task_mpi::TestTaskMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  auto result = ConvertOutputToImage(out, kN);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    EXPECT_EQ(expected, result);
  }
}
