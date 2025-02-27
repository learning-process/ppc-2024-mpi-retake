// Anikin Maksim 2025
#include "seq/anikin_m_graham_scan/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

bool anikin_m_graham_scan_seq::Cmp(Pt a, Pt b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }

bool anikin_m_graham_scan_seq::Cw(Pt a, Pt b, Pt c) {
  return a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y) < 0;
}

bool anikin_m_graham_scan_seq::Ccw(Pt a, Pt b, Pt c) {
  return a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y) > 0;
}

void anikin_m_graham_scan_seq::ConvexHull(std::vector<Pt> &a) {
  if (a.size() <= 1) {
    return;
  }
  std::ranges::sort(a, &Cmp);
  Pt p1 = a[0];
  Pt p2 = a.back();
  std::vector<Pt> up;
  std::vector<Pt> down;
  up.push_back(p1);
  down.push_back(p1);
  for (size_t i = 1; i < a.size(); ++i) {
    if (i == a.size() - 1 || Cw(p1, a[i], p2)) {
      while (up.size() >= 2 && !Cw(up[up.size() - 2], up[up.size() - 1], a[i])) {
        up.pop_back();
      }
      up.push_back(a[i]);
    }
    if (i == a.size() - 1 || Ccw(p1, a[i], p2)) {
      while (down.size() >= 2 && !Ccw(down[down.size() - 2], down[down.size() - 1], a[i])) {
        down.pop_back();
      }
      down.push_back(a[i]);
    }
  }
  a.clear();
  for (size_t i = 0; i < up.size(); ++i) {
    a.push_back(up[i]);
  }
  for (size_t i = down.size() - 2; i > 0; --i) {
    a.push_back(down[i]);
  }
}

bool anikin_m_graham_scan_seq::TestTaskSequential::ValidationImpl() { return task_data->inputs[0] != nullptr; }

bool anikin_m_graham_scan_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<Pt *>(task_data->inputs[0]);
  data_ = std::vector<Pt>(in_ptr, in_ptr + input_size);
  return true;
}

bool anikin_m_graham_scan_seq::TestTaskSequential::RunImpl() {
  ConvexHull(data_);
  return true;
}

bool anikin_m_graham_scan_seq::TestTaskSequential::PostProcessingImpl() {
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(data_.data()));
  task_data->outputs_count.emplace_back(data_.size());
  return true;
}
