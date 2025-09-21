#include <iostream>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "core.hpp"

namespace nb = nanobind;

using const_pyarr_t = nb::ndarray<const double, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using pyarr_t = nb::ndarray<double, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

NB_MODULE(_fastmp, m) {
    m.def("compute_mean_std", [](const_pyarr_t T, uint64_t m) {
        uint64_t n = T.shape(0);
        std::vector<double> mu(n - m + 1);
        std::vector<double> sigma(n - m + 1);

        compute_mean_std(T.data(), mu.data(), sigma.data(), n, m);

        return pyarr_t(mu.data(), {n - m + 1}).cast();
    });
}
