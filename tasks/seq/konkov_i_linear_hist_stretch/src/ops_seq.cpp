#include "seq/konkov_i_linear_hist_stretch/include/ops_seq.hpp"

#include <algorithm>

namespace konkov_i_linear_hist_stretch {

LinearHistogramStretch::LinearHistogramStretch(int image_size, int* image_data)
    : image_size_(image_size), image_data_(image_data) {}

bool LinearHistogramStretch::Validation() const { return image_size_ > 0 && image_data_ != nullptr; }

bool LinearHistogramStretch::PreProcessing() {
  if (!Validation()) return false;

  CalculateGlobalMinMax();
  return true;
}

bool LinearHistogramStretch::Run() {
  StretchPixels();
  return true;
}

bool LinearHistogramStretch::PostProcessing() { return true; }

void LinearHistogramStretch::CalculateGlobalMinMax() {
  global_min_ = *std::min_element(image_data_, image_data_ + image_size_);
  global_max_ = *std::max_element(image_data_, image_data_ + image_size_);
}

void LinearHistogramStretch::StretchPixels() {
  if (global_max_ - global_min_ == 0) {
    return;
  }

  for (int i = 0; i < image_size_; ++i) {
    image_data_[i] =
        static_cast<int>((static_cast<double>(image_data_[i] - global_min_) / (global_max_ - global_min_)) * 255.0);
  }
}

}  // namespace konkov_i_linear_hist_stretch