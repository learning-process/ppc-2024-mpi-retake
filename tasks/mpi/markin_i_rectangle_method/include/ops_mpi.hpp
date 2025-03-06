#pragma once

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <cmath>
#include <limits>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace markin_i_rectangle_mpi {

inline double f(double x) {
    return x * x;
}

class RectangleMpiTask : public ppc::core::Task {
public:
    RectangleMpiTask(ppc::core::TaskDataPtr task_data, boost::mpi::communicator world);

    bool ValidationImpl() override;
    bool PreProcessingImpl() override;
    bool RunImpl() override;
    bool PostProcessingImpl() override;

private:
    double a_;
    double b_;
    int n_;
    double h_;
    double global_sum_;
    boost::mpi::communicator world_;
    int world_rank_;
    int world_size_;
};