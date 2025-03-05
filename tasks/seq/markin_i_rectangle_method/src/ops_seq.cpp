#include "seq\markin_i_rectangle_method\include\ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

float markin_i_rectangle_method_seq::f(float x) {
  return (x * x * x) / 4;
}

bool markin_i_rectangle_method_seq::RectangleSequential::PreProcessingImpl() {
  return true;
}

bool markin_i_rectangle_method_seq::RectangleSequential::ValidationImpl() {
  left_ = *reinterpret_cast<float*>(task_data->inputs[0]);
  right_ = *reinterpret_cast<float*>(task_data->inputs[1]);
  steps_ = *reinterpret_cast<int*>(task_data->inputs[2]);

  output_ = 0;
  return output_ == 0 && left_ < right_ && steps_ > 0;
}

bool markin_i_rectangle_method_seq::RectangleSequential::RunImpl() {
  float step = (right_ - left_) / steps_;
  for (float i = left_ + (step/2); i < right_;i+=step){
    output_+=f(i)*step;
  }
  return true;
}

bool markin_i_rectangle_method_seq::RectangleSequential::PostProcessingImpl() {
  float* output_ptr = reinterpret_cast<float*>(task_data->outputs[0]);
  *output_ptr = output_;
  return true;
}