#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "ersoz_b_horizontal_linear_filtering_gauss/include/ops_mpi.hpp"

TEST(ersoz_b_test_task_mpi, test_gaussian_filter_small) {
  constexpr int N = 16;  // Image is N x N
  std::vector<char> in(N * N, 0);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) in[i * N + j] = static_cast<char>((i + j) % 256);

  std::vector<std::vector<char>> image;
  for (int i = 0; i < N; i++) image.push_back(std::vector<char>(in.begin() + i * N, in.begin() + (i + 1) * N));

  auto sequentialFilter = [&image](double sigma) -> std::vector<std::vector<char>> {
    int Y = static_cast<int>(image.size());
    int X = static_cast<int>(image[0].size());
    std::vector<std::vector<char>> res;
    for (int y = 1; y < Y - 1; y++) {
      std::vector<char> line;
      for (int x = 1; x < X - 1; x++) {
        double brightness = 0;
        for (int i = -1; i <= 1; i++) {
          for (int j = -1; j <= 1; j++) {
            brightness += 1 / (2 * M_PI * sigma * sigma) * exp(-(i * i + j * j) / (2 * sigma * sigma)) *
                          static_cast<int>(image[y + i][x + j]);
          }
        }
        line.push_back(static_cast<char>(brightness));
      }
      res.push_back(line);
    }
    return res;
  };
  auto expected = sequentialFilter(0.5);

  std::vector<char> out((N - 2) * (N - 2), 0);
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

  std::vector<std::vector<char>> result;
  for (int i = 0; i < N - 2; i++)
    result.push_back(std::vector<char>(out.begin() + i * (N - 2), out.begin() + (i + 1) * (N - 2)));

  EXPECT_EQ(expected, result);
}
