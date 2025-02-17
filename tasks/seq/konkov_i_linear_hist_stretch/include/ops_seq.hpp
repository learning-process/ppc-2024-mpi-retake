#ifndef MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_SEQ_HPP_
#define MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_SEQ_HPP

namespace konkov_i_linear_hist_stretch {

class LinearHistogramStretch {
 public:
  explicit LinearHistogramStretch(int image_size, int* image_data);
  [[nodiscard]] bool Validation() const;
  bool PreProcessing();
  bool Run();
  static bool PostProcessing();

 private:
  int image_size_;
  int* image_data_;
  int global_min_;
  int global_max_;

  void CalculateGlobalMinMax();
  void StretchPixels();
};

}  // namespace konkov_i_linear_hist_stretch

#endif  // MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_SEQ_HPP_