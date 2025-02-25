// Anikin Maksim 2025
#include "seq/anikin_m_graham_scan/include/ops_seq.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

bool anikin_m_graham_scan_seq::Cmp(Pt a, Pt b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }

bool anikin_m_graham_scan_seq::Cw(Pt a, Pt b, Pt c) {
  return a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y) < 0;
}

bool anikin_m_graham_scan_seq::Ccw(Pt a, Pt b, Pt c) {
  return a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y) > 0;
}

bool anikin_m_graham_scan_seq::TestData(std::vector<Pt> alg_out_, int case_) {
  // case 0
  //  all_points  = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 1
  //  all_points  = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (4, 4), (0, 4)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 2
  //  all_points  = [(0, 0), (1, 3), (2, 1), (3, 2), (4, 0), (2, 4)]
  //  hull_points = [(0, 0), (4, 0), (2, 4), (1, 3)]
  bool out = true;
  switch (case_) {
    case 1:
    case 0:
      out &= (alg_out_.size() == 4);

      out &= (alg_out_[0].x == 0);
      out &= (alg_out_[0].y == 0);

      out &= (alg_out_[1].x == 0);
      out &= (alg_out_[1].y == 4);

      out &= (alg_out_[2].x == 4);
      out &= (alg_out_[2].y == 4);

      out &= (alg_out_[3].x == 4);
      out &= (alg_out_[3].y == 0);
      break;
    case 2:
      out &= (alg_out_.size() == 4);

      out &= (alg_out_[0].x == 0);
      out &= (alg_out_[0].y == 0);

      out &= (alg_out_[1].x == 1);
      out &= (alg_out_[1].y == 3);

      out &= (alg_out_[2].x == 2);
      out &= (alg_out_[2].y == 4);

      out &= (alg_out_[3].x == 4);
      out &= (alg_out_[3].y == 0);
      break;
    default:
      break;
  }
  return out;
}

void anikin_m_graham_scan_seq::CreateTestData(std::vector<Pt> &alg_in_, int case_) {
  // case 0
  //  all_points  = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 1
  //  all_points  = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (4, 4), (0, 4)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 2
  //  all_points  = [(0, 0), (1, 3), (2, 1), (3, 2), (4, 0), (2, 4)]
  //  hull_points = [(0, 0), (4, 0), (2, 4), (1, 3)]
  alg_in_.clear();
  switch (case_) {
    case 0:
      alg_in_.push_back({0, 0});
      alg_in_.push_back({4, 0});
      alg_in_.push_back({4, 4});
      alg_in_.push_back({0, 4});
      alg_in_.push_back({2, 2});
      break;
    case 1:
      alg_in_.push_back({0, 0});
      alg_in_.push_back({1, 1});
      alg_in_.push_back({2, 2});
      alg_in_.push_back({3, 3});
      alg_in_.push_back({4, 0});
      alg_in_.push_back({4, 4});
      alg_in_.push_back({0, 4});
      break;
    case 2:
      alg_in_.push_back({0, 0});
      alg_in_.push_back({1, 3});
      alg_in_.push_back({2, 1});
      alg_in_.push_back({3, 2});
      alg_in_.push_back({4, 0});
      alg_in_.push_back({2, 4});
      break;
    default:
      break;
  }
}

void anikin_m_graham_scan_seq::ConvexHull(std::vector<Pt> &a) {
  if (a.size() == 1 || a.size() == 0) {
    return;
  }
  sort(a.begin(), a.end(), &Cmp);
  Pt p1 = a[0], p2 = a.back();
  std::vector<Pt> up, down;
  up.push_back(p1);
  down.push_back(p1);
  for (size_t i = 1; i < a.size(); ++i) {
    if (i == a.size() - 1 || Cw(p1, a[i], p2)) {
      while (up.size() >= 2 && !Cw(up[up.size() - 2], up[up.size() - 1], a[i])) up.pop_back();
      up.push_back(a[i]);
    }
    if (i == a.size() - 1 || Ccw(p1, a[i], p2)) {
      while (down.size() >= 2 && !Ccw(down[down.size() - 2], down[down.size() - 1], a[i])) down.pop_back();
      down.push_back(a[i]);
    }
  }
  a.clear();
  for (size_t i = 0; i < up.size(); ++i) a.push_back(up[i]);
  for (size_t i = down.size() - 2; i > 0; --i) a.push_back(down[i]);
}

void anikin_m_graham_scan_seq::CreateRandomData(std::vector<Pt> &alg_in_, int count) {
  alg_in_.clear();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 100.0);
  Pt rand_;
  for (int i = 0; i < count; i++) {
    rand_.x = dis(gen);
    rand_.y = dis(gen);
    alg_in_.push_back(rand_);
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
