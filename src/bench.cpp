#include <vector>

#include "fastmp.hpp"

int main()
{
    size_t n = 7200, m = 10;

    std::vector<double> T(n), P(n - m + 1);

    for (int i = 0; i < 50; i++) {
        stomp(T.data(), P.data(), n, m);
    }
}
