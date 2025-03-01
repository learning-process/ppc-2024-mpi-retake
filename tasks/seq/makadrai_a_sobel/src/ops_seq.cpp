#include "seq/makadrai_a_sobel/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

bool makadrai_a_sobel_seq::Sobel::PreProcessingImpl() {
  height_img_ = (int)task_data->inputs_count[1];
  width_img_ = (int)task_data->inputs_count[0];

  img_.resize((width_img_ + peding_) * (height_img_ + peding_));
  simg_.resize(width_img_ * height_img_, 0);

  const auto* in = reinterpret_cast<int*>(task_data->inputs[0]);

  for (int i = 0; i < height_img_; i++) {
    std::copy(in + (i * width_img_), in + ((i + 1) * width_img_),
              img.begin() + (((i + 1) * (width_img_ + peding_)) + 1));
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
  for (int i = 1; i < height_img_ + 1; i++) {
    for (int j = 1; j < width_img_ + 1; j++) {
      int g_x = (-1 * img_[((i - 1) * (width_img_ + peding_)) + (j - 1)]) +
                (-1 * img_[((i - 1) * (width_img_ + peding_)) + (j + 1)]) -
                (2 * img_[((i - 1) * (width_img_ + peding_)) + j]) +
                (2 * img_[((i + 1) * (width_img_ + peding_)) + j]) +
                (1 * img_[((i + 1) * (width_img_ + peding_)) + (j - 1)]) +
                1 * img_[((i + 1) * (width_img_ + peding_)) + (j + 1)];

      int g_y = (1 * img_[((i - 1) * (width_img_ + peding_)) + (j - 1)]) +
                (2 * img_[(i * (width_img_ + peding_)) + (j - 1)]) +
                (1 * img_[((i + 1) * (width_img_ + peding_)) + (j - 1)]) +
                (-1 * img_[((i - 1) * (width_img_ + peding_)) + (j + 1)]) -
                (2 * img_[(i * (width_img_ + peding_)) + (j + 1)]) + 
                -1 * img_[(i + 1) * (width_img_ + peding_) + (j + 1)];

      int temp = (int)std::sqrt(std::pow(g_x, 2) + std::pow(g_y, 2));
      max_z = std::max(max_z, temp);
      simg_[((i - 1) * width_img_) + (j - 1)] = temp;
    }
  }

  for (int i = 0; i < width_img_; i++) {
    for (int j = 0; j < height_img_; j++) {
      simg_[i * height_img_ + j] = (int)((double)simg_[(i * height_img_) + j] / max_z) * 255;
    }
  }

  return true;
}

bool makadrai_a_sobel_seq::Sobel::PostProcessingImpl() {
  std::ranges::copy(simg_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
