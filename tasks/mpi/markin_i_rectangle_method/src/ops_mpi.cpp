#include "mpi/markin_i_rectangle_method/include/ops_mpi.hpp"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/serialization/vector.hpp>
namespace markin_i_rectangle_mpi {

RectangleMpiTask::RectangleMpiTask(ppc::core::TaskDataPtr task_data, boost::mpi::communicator world)
    : Task(task_data),
      world_(world),
      a_(0.0),
      b_(1.0),
      n_(1000000),
      h_(0.0),
      global_sum_(0.0),
      world_rank_(0),
      world_size_(0)
       {}

bool RectangleMpiTask::ValidationImpl() {
    if (world_.rank() == 0) {
        if (task_data->inputs.size() != 3 || task_data->outputs.size() != 1) {
            std::cerr << "Ошибка: Неверное количество входных или выходных данных.\n";
            std::cout << "SizeI: " << task_data->inputs.size() << std::endl;
            std::cout << "SizeO: " << task_data->outputs.size() << std::endl;
            return false;
        }
    }
    return true;
}

bool RectangleMpiTask::PreProcessingImpl() {
    world_size_ = world_.size();
    world_rank_ = world_.rank();

    if (world_.rank() == 0) {
        double* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
        double* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
        int* n_ptr = reinterpret_cast<int*>(task_data->inputs[2]);

        if (!a_ptr || !b_ptr || !n_ptr) {
            std::cerr << "Ошибка: Неверные указатели на входные данные.\n";
            return false;
        }
        a_ = *a_ptr;
        b_ = *b_ptr;
        n_ = *n_ptr;
        h_ = (b_ - a_) / n_;
    }

    boost::mpi::broadcast(world_, a_, 0);
    boost::mpi::broadcast(world_, b_, 0);
    boost::mpi::broadcast(world_, n_, 0);
    h_ = (b_ - a_) / n_;
    return true;
}

bool RectangleMpiTask::RunImpl() {
    int local_n = n_ / world_size_;
    int remainder = n_ % world_size_;
    int start_index, end_index;

    if (world_rank_ < remainder) {
        local_n++;
        start_index = world_rank_ * local_n;
        end_index = start_index + local_n;
    } else {
        start_index = remainder * (local_n + 1) + (world_rank_ - remainder) * local_n;
        end_index = start_index + local_n;
    }

    double local_sum = 0.0;
    for (int i = start_index; i < end_index; ++i) {
        double x_i = a_ + (i + 0.5) * h_;
        local_sum += f(x_i) * h_;
    }

    boost::mpi::reduce(world_, local_sum, global_sum_, std::plus<double>(), 0);

    return true;
}

bool RectangleMpiTask::PostProcessingImpl() {
    if (world_.rank() == 0) {
        double* out = reinterpret_cast<double*>(task_data->outputs[0]);
        *out = global_sum_;
    }
     boost::mpi::broadcast(world_, global_sum_,0);
    return true;
}

} // namespace markin_i_rectangle_mpi