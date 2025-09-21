#include <nanobind/nanobind.h>

int add(int a, int b) { return a + b; }

NB_MODULE(_fastmp, m) {
    m.def("add", &add);
}
