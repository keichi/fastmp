#include <cmath>

void compute_mean_std(const double *T, double *mu, double *sigma, size_t n, size_t m)
{
    double sum_T = 0.0;
    double sum_T_lag = 0.0;
    double sum_T2 = 0.0;
    double sum_T2_lag = 0.0;

    for (size_t i = 0; i < n; i++) {
        sum_T += T[i];
        sum_T2 += T[i] * T[i];

        if (i + 1 >= m) {
            mu[i + 1 - m] = (sum_T - sum_T_lag) / m;
            sigma[i + 1 - m] = std::sqrt((sum_T2 - sum_T2_lag) / m - mu[i + 1 - m] * mu[i + 1 - m]);

            sum_T_lag += T[i + 1 - m];
            sum_T2_lag += T[i + 1 - m] * T[i + 1 - m];
        }
    }
}

