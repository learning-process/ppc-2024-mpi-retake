#include "seq/makadrai_a_sobel/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool makadrai_a_sobel_seq::Sobel::PreProcessingImpl() {
  height_img = task_data->inputs_count[1];
  width_img = task_data->inputs_count[0];

  img.resize((width_img + peding) * (height_img + peding));
  simg.resize(width_img * height_img, 0);

  const auto* in = reinterpret_cast<int*>(task_data->inputs[0]);

  for (int i = 0; i < height_img; i++) {
    std::copy(in + (i * width_img), in + ((i + 1) * width_img), img.begin() + ((i + 1) * (width_img + peding) + 1));
  }
  return true;
}

bool makadrai_a_sobel_seq::Sobel::ValidationImpl() {
  return task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->outputs_count[1] == task_data->inputs_count[1] && task_data->inputs_count[0] > 0 &&
         task_data->inputs_count[1] > 0;
}

bool makadrai_a_sobel_seq::Sobel::RunImpl() {
  int max_z = 1;
  for (int i = 1; i < height_img + 1; i++) {
    for (int j = 1; j < width_img + 1; j++) {
      int G_x = -1 * img[(i - 1) * (width_img + peding) + (j - 1)] +
                -1 * img[(i - 1) * (width_img + peding) + (j + 1)] - 2 * img[(i - 1) * (width_img + peding) + j] +
                2 * img[(i + 1) * (width_img + peding) + j] + 1 * img[(i + 1) * (width_img + peding) + (j - 1)] +
                1 * img[(i + 1) * (width_img + peding) + (j + 1)];

      int G_y = 1 * img[(i - 1) * (width_img + peding) + (j - 1)] + 2 * img[i * (width_img + peding) + (j - 1)] +
                1 * img[(i + 1) * (width_img + peding) + (j - 1)] + -1 * img[(i - 1) * (width_img + peding) + (j + 1)] -
                2 * img[i * (width_img + peding) + (j + 1)] + -1 * img[(i + 1) * (width_img + peding) + (j + 1)];

      int temp = std::sqrt(std::pow(G_x, 2) + std::pow(G_y, 2));
      max_z = std::max(max_z, temp);
      simg[(i - 1) * width_img + (j - 1)] = temp;
    }
  }

  for (int i = 0; i < width_img; i++) {
    for (int j = 0; j < height_img; j++) {
      simg[i * height_img + j] = ((double)simg[i * height_img + j] / max_z) * 255;
    }
  }

  return true;
}

bool makadrai_a_sobel_seq::Sobel::PostProcessingImpl() {
  std::copy(simg.begin(), simg.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
