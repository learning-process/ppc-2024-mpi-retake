#ifndef MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_MPI_HPP
#define MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_MPI_HPP

namespace konkov_i_linear_hist_stretch {

class LinearHistogramStretch {
 public:
  explicit LinearHistogramStretch(int image_size, int* image_data);
  ~LinearHistogramStretch();
  [[nodiscard]] bool Validation() const;
  bool PreProcessing();
  bool Run();
  bool PostProcessing();

 private:
  int image_size_;
  int* image_data_;
  int rank_;
  int size_;
  int local_size_;
  int* local_data_;

  void DistributeData();
  void GatherData();
};

}  // namespace konkov_i_linear_hist_stretch

#endif  // MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_MPI_HPP