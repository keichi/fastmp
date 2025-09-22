#include <cmath>
#include <iostream>
#include <vector>

#include "pocketfft.hpp"

void sliding_window_dot_prodouct(const double *T, const double *Q, double *QT, size_t n, size_t m)
{
    std::vector<double> Ta(n * 2), Qra(n * 2);
    std::vector<std::complex<double>> Taf(n + 1), Qraf(n + 1);
    
    for (size_t i = 0; i < n; i++) {
        Ta[i] = T[i];
    }

    for (size_t i = 0; i < m; i++) {
        Qra[i] = Q[m - i - 1];
    }

    pocketfft::r2c({n * 2}, {sizeof(double)}, {sizeof(std::complex<double>)}, 0, true,
                   Qra.data(), Qraf.data(), 1.0);

    pocketfft::r2c({n * 2}, {sizeof(double)}, {sizeof(std::complex<double>)}, 0, true,
                   Ta.data(), Taf.data(), 1.0);

    for (size_t i = 0; i < n + 1; i++) {
        Qraf[i] *= Taf[i];
    }

    pocketfft::c2r({n * 2}, {sizeof(std::complex<double>)}, {sizeof(double)}, 0, false,
                   Qraf.data(), Qra.data(), 1.0 / (n * 2));

    for (size_t i = m - 1; i < n; i++) {
        QT[i - m + 1] = Qra[i];
    }
}

void compute_mean_std(const double *T, double *mu, double *sigma, size_t n, size_t m)
{
    double sum_T = 0.0;
    double sum_T_lag = 0.0;
    double sum_T2 = 0.0;
    double sum_T2_lag = 0.0;

    for (size_t i = 0; i < n; i++) {
        sum_T += T[i];
        sum_T2 += T[i] * T[i];

        if (i - m >= 0) {
            sum_T_lag += T[i - m];
            sum_T2_lag += T[i - m] * T[i - m];
        }

        mu[i - m + 1] = (sum_T - sum_T_lag) / m;
        sigma[i - m + 1] = std::sqrt((sum_T2 - sum_T2_lag) / m - mu[i - m + 1] * mu[i - m + 1]);
    }
}

// calc_dist_profile()
// {
//     return np.sqrt(2.0 * m * (1.0 - (QT - m * mu[i] * mu) / (m * sigma[i] * sigma)))
// }
// 
// stomp()
// {
//     for (int i = 1; i < n - m + 1; i++) {
//         for (int j = n - m; j > 0; j--) {
//             QT[j] = QT[j - 1] - T[j - 1] * T[i - 1] + T[j + m - 1] * T[i + m - 1];
//         }
//         QT[0] = QT_first[i];
// 
//         D = calc_dist_profile(QT, m, mu, sigma, i):
// 
//         // Apply exclusion zone
//         for j in range(i - excl_zone, i + excl_zone):
//             if 0 <= j < D.shape[0]:
//                 D[j] = np.inf
// 
//         P = np.minimum(P, D)
//     }
// }

