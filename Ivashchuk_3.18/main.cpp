#include "../../../modules/task_3/ivashchuk_3.18/ivashchuk_3.18.h"

std::vector<double> Get_Random_Vector(int size) {
    std::mt19937 gen(time(0));
    std::uniform_real_distribution<> urd(0, 10000);
    std::vector<double> vector(size);
    for (int i = 0; i < static_cast<int>(vector.size()); i++) {
        vector[i] = urd(gen);
    }
    return vector;
}

int LeftOfThePoint(double num) {
    int num_of_radix = 0;
    while (num > 1) {
        num /= 10;
        num_of_radix++;
    }
    return num_of_radix;
}

int RightOfThePoint(double num) {
    std::ostringstream strs;
    strs << num;
    std::string str = strs.str();
    if (str.find('.')) {
        return -(static_cast<int>(str.find('.')) -
            static_cast<int>(str.size())) - 1;
    }
    else {
        return 0;
    }
}

std::vector<double> Merge(const std::vector<double>& vec_left,
    const std::vector<double>& vec_right) {
    std::vector<double> result((vec_left.size() + vec_right.size()));

    int i = 0, j = 0, k = 0;
    while (i < static_cast<int>(vec_left.size())
        && j < static_cast<int>(vec_right.size())) {
        if (vec_left[i] < vec_right[j])
            result[k] = vec_left[i++];
        else
            result[k] = vec_right[j++];
        k++;
    }
    while (i < static_cast<int>(vec_left.size())) {
        result[k++] = vec_left[i++];
    }
    while (j < static_cast<int>(vec_right.size())) {
        result[k++] = vec_right[j++];
    }
    return result;
}


int GetDigit(double num, int radix) {
    if (radix > 0) {
        double mask = pow(10, radix);
        double tmp = num / mask;
        return static_cast<int>(tmp) % 10;
    }
    return  static_cast<int>(num * pow(10, -radix)) % 10;
}

std::vector<double> RadixSort(const std::vector<double>& vect, int rad) {
    std::vector<double> res;
    std::vector <std::vector<double>> radix(10);

    for (int i = 0; i < static_cast<int>(vect.size()); i++) {
        radix[GetDigit(vect[i], rad)].push_back(vect[i]);
    }
    for (int i = 0; i < static_cast<int>(radix.size()); ++i)
        for (int j = 0; j < static_cast<int>(radix[i].size()); ++j)
            res.push_back(radix[i][j]);
    return res;
}

std::vector<double> Not_Parallel_Radix_Sort(const std::vector<double>& vect) {
    int radixNegativeZero = 0;
    int maxRadixNegativeZero = RightOfThePoint(vect[0]);
    for (int i = 1; i < static_cast<int>(vect.size()); ++i) {
        radixNegativeZero = RightOfThePoint(vect[i]);
        if (radixNegativeZero > maxRadixNegativeZero) {
            maxRadixNegativeZero = radixNegativeZero;
        }
    }
    double max = vect[0];
    for (int i = 1; i < static_cast<int>(vect.size()); i++) {
        if (vect[i] > max) {
            max = vect[i];
        }
    }
    int maxRadixPositiveZero = LeftOfThePoint(max);
    std::vector<double> result(vect);
    for (int i = -maxRadixNegativeZero; i <= maxRadixPositiveZero; i++) {
        result = RadixSort(result, i);
    }
    return result;
}

std::vector<double> Parallel_Radix_Sort(const std::vector<double>& vec) {
    int ProcNum, ProcRank;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    std::vector<double> result;

    int size = vec.size();
    std::vector<int> sendcounts(ProcNum);
    std::vector<int> displs(ProcNum);

    int count;

    if ((ProcRank < size % ProcNum))
        count = (size / ProcNum) + 1;
    else
        count = size / ProcNum;

    std::vector<double> recvbuf(count);

    if (recvbuf.size() == 0)
        recvbuf.resize(1);

    displs[0] = 0;

    for (int i = 0; i < ProcNum; i++) {
        if (i < (size % ProcNum))
            sendcounts[i] = (size / ProcNum) + 1;
        else
            sendcounts[i] = size / ProcNum;
        if (i > 0)
            displs[i] = (displs[i - 1] + sendcounts[i - 1]);
    }

    MPI_Scatterv(vec.data(), sendcounts.data(), displs.data(),
        MPI_DOUBLE, recvbuf.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    recvbuf = Not_Parallel_Radix_Sort(recvbuf);

    if (ProcRank != 0) {
        MPI_Send(recvbuf.data(), count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else {
        result = recvbuf;
        MPI_Status status;
        for (int i = 1; i < ProcNum; i++) {
            recvbuf.resize(sendcounts[i]);
            MPI_Recv(recvbuf.data(), sendcounts[i], MPI_DOUBLE,
                i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            result = Merge(result, recvbuf);
        }
    }
    return result;
}