// Anikin Maksim 2025
#include "mpi/anikin_m_graham_scan/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

bool anikin_m_graham_scan_mpi::Cmp(const Pt& a, const Pt& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }

bool anikin_m_graham_scan_mpi::Cw(const Pt& a, const Pt& b, const Pt& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) < 0;
}

bool anikin_m_graham_scan_mpi::Ccw(const Pt& a, const Pt& b, const Pt& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
}

void anikin_m_graham_scan_mpi::ConvexHull(std::vector<Pt>& points) {
  if (points.size() <= 1) {
    return;
  }
  std::ranges::sort(points, &Cmp);
  Pt p1 = points[0];
  Pt p2 = points.back();
  std::vector<Pt> up;
  std::vector<Pt> down;
  up.push_back(p1);
  down.push_back(p1);
  for (size_t i = 1; i < points.size(); ++i) {
    if (i == points.size() - 1 || Cw(p1, points[i], p2)) {
      while (up.size() >= 2 && !Cw(up[up.size() - 2], up.back(), points[i])) {
        up.pop_back();
      }
      up.push_back(points[i]);
    }
    if (i == points.size() - 1 || Ccw(p1, points[i], p2)) {
      while (down.size() >= 2 && !Ccw(down[down.size() - 2], down.back(), points[i])) {
        down.pop_back();
      }
      down.push_back(points[i]);
    }
  }
  points.clear();
  points.insert(points.end(), up.begin(), up.end());
  for (size_t i = down.size() - 2; i > 0; --i) {
    points.push_back(down[i]);
  }
}

bool anikin_m_graham_scan_mpi::TestData(std::vector<Pt> alg_out, int test) {
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
  switch (test) {
    case 1:
    case 0:
      out &= (alg_out.size() == 4);

      out &= (alg_out[0].x == 0);
      out &= (alg_out[0].y == 0);

      out &= (alg_out[1].x == 0);
      out &= (alg_out[1].y == 4);

      out &= (alg_out[2].x == 4);
      out &= (alg_out[2].y == 4);

      out &= (alg_out[3].x == 4);
      out &= (alg_out[3].y == 0);
      break;
    case 2:
      out &= (alg_out.size() == 4);

      out &= (alg_out[0].x == 0);
      out &= (alg_out[0].y == 0);

      out &= (alg_out[1].x == 1);
      out &= (alg_out[1].y == 3);

      out &= (alg_out[2].x == 2);
      out &= (alg_out[2].y == 4);

      out &= (alg_out[3].x == 4);
      out &= (alg_out[3].y == 0);
      break;
    default:
      break;
  }
  return out;
}

void anikin_m_graham_scan_mpi::CreateTestData(std::vector<Pt>& alg_in, int test) {
  // case 0
  //  all_points  = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 1
  //  all_points  = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (4, 4), (0, 4)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 2
  //  all_points  = [(0, 0), (1, 3), (2, 1), (3, 2), (4, 0), (2, 4)]
  //  hull_points = [(0, 0), (4, 0), (2, 4), (1, 3)]
  alg_in.clear();
  switch (test) {
    case 0:
      alg_in.push_back({0, 0});
      alg_in.push_back({4, 0});
      alg_in.push_back({4, 4});
      alg_in.push_back({0, 4});
      alg_in.push_back({2, 2});
      break;
    case 1:
      alg_in.push_back({0, 0});
      alg_in.push_back({1, 1});
      alg_in.push_back({2, 2});
      alg_in.push_back({3, 3});
      alg_in.push_back({4, 0});
      alg_in.push_back({4, 4});
      alg_in.push_back({0, 4});
      break;
    case 2:
      alg_in.push_back({0, 0});
      alg_in.push_back({1, 3});
      alg_in.push_back({2, 1});
      alg_in.push_back({3, 2});
      alg_in.push_back({4, 0});
      alg_in.push_back({2, 4});
      break;
    default:
      break;
  }
}

void anikin_m_graham_scan_mpi::CreateRandomData(std::vector<Pt>& alg_in, int count) {
  alg_in.clear();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 100.0);
  Pt rand;
  for (int i = 0; i < count; i++) {
    rand.x = (int)dis(gen);
    rand.y = (int)dis(gen);
    alg_in.push_back(rand);
  }
}

bool anikin_m_graham_scan_mpi::TestTaskMPI::ValidationImpl() { return task_data->inputs[0] != nullptr; }

bool anikin_m_graham_scan_mpi::TestTaskMPI::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Pt*>(task_data->inputs[0]);
  data_ = std::vector<Pt>(in_ptr, in_ptr + input_size);
  return true;
}

bool anikin_m_graham_scan_mpi::TestTaskMPI::RunImpl() {
  MPI_Datatype mpi_pt;  // NOLINT
  MPI_Type_contiguous(2, MPI_INT, &mpi_pt);
  MPI_Type_commit(&mpi_pt);

  std::vector<Pt> local_points;
  int n = 0;
  if (world_.rank() == 0) {
    n = (int)data_.size();
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, world_);
  int local_count = (n / world_.size()) + ((world_.rank() < n % world_.size()) ? 1 : 0);
  local_points.resize(local_count);

  int world_size = world_.size();
  int* counts = new int[world_size];
  int* displs = new int[world_size];
  int offset = 0;
  for (int i = 0; i < world_.size(); i++) {
    counts[i] = n / world_.size() + (i < n % world_.size() ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }

  MPI_Scatterv(world_.rank() == 0 ? data_.data() : nullptr, counts, displs, mpi_pt, local_points.data(), local_count,
               mpi_pt, 0, world_);

  Pt local_p1 = {.x = 0, .y = 0};
  Pt local_p2 = {.x = 0, .y = 0};
  if (!local_points.empty()) {
    local_p1 = *std::ranges::min_element(local_points, Cmp);
    local_p2 = *std::ranges::max_element(local_points, Cmp);
  }

  std::vector<Pt> all_p1(world_.size());
  std::vector<Pt> all_p2(world_.size());
  MPI_Gather(&local_p1, 1, mpi_pt, all_p1.data(), 1, mpi_pt, 0, world_);
  MPI_Gather(&local_p2, 1, mpi_pt, all_p2.data(), 1, mpi_pt, 0, world_);

  Pt global_p1;
  Pt global_p2;
  if (world_.rank() == 0) {
    global_p1 = *std::ranges::min_element(all_p1, Cmp);
    global_p2 = *std::ranges::max_element(all_p2, Cmp);
  }

  MPI_Bcast(&global_p1, 1, mpi_pt, 0, world_);
  MPI_Bcast(&global_p2, 1, mpi_pt, 0, world_);

  local_points.push_back(global_p1);
  local_points.push_back(global_p2);
  std::ranges::sort(local_points, Cmp);
  auto last = std::ranges::unique(local_points);
  local_points.erase(last.begin(), local_points.end());

  ConvexHull(local_points);

  if (world_.rank() != 0) {
    int size = (int)local_points.size();
    MPI_Send(&size, 1, MPI_INT, 0, 0, world_);
    MPI_Send(local_points.data(), size, mpi_pt, 0, 0, world_);
  } else {
    std::vector<Pt> final_hull = local_points;
    for (int i = 1; i < world_.size(); ++i) {
      int recv_size = 0;
      MPI_Recv(&recv_size, 1, MPI_INT, i, 0, world_, MPI_STATUS_IGNORE);
      std::vector<Pt> temp(recv_size);
      MPI_Recv(temp.data(), recv_size, mpi_pt, i, 0, world_, MPI_STATUS_IGNORE);
      final_hull.insert(final_hull.end(), temp.begin(), temp.end());
    }
    ConvexHull(final_hull);
    data_.clear();
    data_.insert(data_.begin(), final_hull.begin(), final_hull.end());
  }

  delete[] counts;
  delete[] displs;
  MPI_Type_free(&mpi_pt);

  return true;
}

bool anikin_m_graham_scan_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_.data()));
    task_data->outputs_count.emplace_back(data_.size());
  }
  return true;
}