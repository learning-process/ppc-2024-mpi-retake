// Anikin Maksim 2025
#include "mpi/anikin_m_graham_scan/include/ops_mpi.hpp"

#include <vector>
#include <random>
#include <mpi.h>

bool anikin_m_graham_scan_mpi::cmp(const pt& a, const pt& b) { 
  return a.x < b.x || (a.x == b.x && a.y < b.y); 
}

bool anikin_m_graham_scan_mpi::cw(const pt& a, const pt& b, const pt& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) < 0;
}

bool anikin_m_graham_scan_mpi::ccw(const pt& a, const pt& b, const pt& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
}

void anikin_m_graham_scan_mpi::convex_hull(std::vector<pt>& points) {
  if (points.size() <= 1) return;
  std::sort(points.begin(), points.end(), &cmp);
  pt p1 = points[0], p2 = points.back();
  std::vector<pt> up, down;
  up.push_back(p1);
  down.push_back(p1);
  for (size_t i = 1; i < points.size(); ++i) {
    if (i == points.size() - 1 || cw(p1, points[i], p2)) {
      while (up.size() >= 2 && !cw(up[up.size() - 2], up.back(), points[i])) up.pop_back();
      up.push_back(points[i]);
    }
    if (i == points.size() - 1 || ccw(p1, points[i], p2)) {
      while (down.size() >= 2 && !ccw(down[down.size() - 2], down.back(), points[i])) down.pop_back();
      down.push_back(points[i]);
    }
  }
  points.clear();
  points.insert(points.end(), up.begin(), up.end());
  for (int i = down.size() - 2; i > 0; --i) points.push_back(down[i]);
}

bool anikin_m_graham_scan_mpi::test_data(std::vector<pt> alg_out_, int case_) {
  // case 0
  //  all_points  = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 1
  //  all_points  = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (4, 4), (0, 4)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 2
  //  all_points  = [(0, 0), (1, 3), (2, 1), (3, 2), (4, 0), (2, 4)]
  //  hull_points = [(0, 0), (4, 0), (2, 4), (1, 3)]
  bool out_ = true;
  switch (case_) {
    case 1:
    case 0:
      out_ &= (alg_out_.size() == 4);

      out_ &= (alg_out_[0].x == 0);
      out_ &= (alg_out_[0].y == 0);

      out_ &= (alg_out_[1].x == 0);
      out_ &= (alg_out_[1].y == 4);

      out_ &= (alg_out_[2].x == 4);
      out_ &= (alg_out_[2].y == 4);

      out_ &= (alg_out_[3].x == 4);
      out_ &= (alg_out_[3].y == 0);
      break;
    case 2:
      out_ &= (alg_out_.size() == 4);

      out_ &= (alg_out_[0].x == 0);
      out_ &= (alg_out_[0].y == 0);

      out_ &= (alg_out_[1].x == 1);
      out_ &= (alg_out_[1].y == 3);

      out_ &= (alg_out_[2].x == 2);
      out_ &= (alg_out_[2].y == 4);

      out_ &= (alg_out_[3].x == 4);
      out_ &= (alg_out_[3].y == 0);
      break;
    default:
      break;
  }
  return out_;
}

void anikin_m_graham_scan_mpi::create_test_data(std::vector<pt>& alg_in_, int case_) {
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

void anikin_m_graham_scan_mpi::create_random_data(std::vector<pt>& alg_in_, int count) {
  alg_in_.clear();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 100.0);
  pt rand_;
  for (int i = 0; i < count; i++) {
    rand_.x = (int) dis(gen);
    rand_.y = (int) dis(gen);
    alg_in_.push_back(rand_);
  }
}

bool anikin_m_graham_scan_mpi::TestTaskMPI::ValidationImpl() {
  return task_data->inputs[0] != nullptr;
}

bool anikin_m_graham_scan_mpi::TestTaskMPI::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<pt*>(task_data->inputs[0]);
  data_ = std::vector<pt>(in_ptr, in_ptr + input_size);
  return true;
}

bool anikin_m_graham_scan_mpi::TestTaskMPI::RunImpl() {
  MPI_Datatype MPI_PT;
  MPI_Type_contiguous(2, MPI_INT, &MPI_PT);
  MPI_Type_commit(&MPI_PT);

  std::vector<pt> local_points;
  int n = 0;
  if (world_.rank() == 0) {
    n = data_.size();
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, world_);
  int local_count = n / world_.size() + ((world_.rank() < n % world_.size()) ? 1 : 0);
  local_points.resize(local_count);

  int world_size_ = world_.size();
  int* counts = new int[world_size_];
  int* displs = new int[world_size_];
  int offset = 0;
  for (int i = 0; i < world_.size(); i++) {
    counts[i] = n / world_.size() + (i < n % world_.size() ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }

  MPI_Scatterv(world_.rank() == 0 ? data_.data() : nullptr, counts, displs, MPI_PT, local_points.data(),
               local_count, MPI_PT, 0, world_);

  pt local_p1 = {0, 0}, local_p2 = {0, 0};
  if (!local_points.empty()) {
    local_p1 = *std::min_element(local_points.begin(), local_points.end(), cmp);
    local_p2 = *std::max_element(local_points.begin(), local_points.end(), cmp);
  }

  std::vector<pt> all_p1(world_.size()), all_p2(world_.size());
  MPI_Gather(&local_p1, 1, MPI_PT, all_p1.data(), 1, MPI_PT, 0, world_);
  MPI_Gather(&local_p2, 1, MPI_PT, all_p2.data(), 1, MPI_PT, 0, world_);

  pt global_p1, global_p2;
  if (world_.rank() == 0) {
    global_p1 = *std::min_element(all_p1.begin(), all_p1.end(), cmp);
    global_p2 = *std::max_element(all_p2.begin(), all_p2.end(), cmp);
  }

  MPI_Bcast(&global_p1, 1, MPI_PT, 0, world_);
  MPI_Bcast(&global_p2, 1, MPI_PT, 0, world_);

  local_points.push_back(global_p1);
  local_points.push_back(global_p2);
  std::sort(local_points.begin(), local_points.end(), cmp);
  auto last = std::unique(local_points.begin(), local_points.end());
  local_points.erase(last, local_points.end());

  convex_hull(local_points);

  if (world_.rank() != 0) {
    int size = local_points.size();
    MPI_Send(&size, 1, MPI_INT, 0, 0, world_);
    MPI_Send(local_points.data(), size, MPI_PT, 0, 0, world_);
  } else {
    std::vector<pt> final_hull = local_points;
    for (int i = 1; i < world_.size(); ++i) {
      int recv_size;
      MPI_Recv(&recv_size, 1, MPI_INT, i, 0, world_, MPI_STATUS_IGNORE);
      std::vector<pt> temp(recv_size);
      MPI_Recv(temp.data(), recv_size, MPI_PT, i, 0, world_, MPI_STATUS_IGNORE);
      final_hull.insert(final_hull.end(), temp.begin(), temp.end());
    }
    convex_hull(final_hull);
    data_.clear();
    data_.insert(data_.begin(), final_hull.begin(), final_hull.end());
  }

  delete[] counts;
  delete[] displs;
  MPI_Type_free(&MPI_PT);

  return true;
}

bool anikin_m_graham_scan_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_.data()));
    task_data->outputs_count.emplace_back(data_.size());
  }
  return true;
}