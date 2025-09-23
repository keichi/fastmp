#include <cmath>
#include <limits>
#include <vector>

#define POCKETFFT_NO_MULTITHREADING
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

        if (i + 1 >= m) {
            sum_T_lag += T[i - m];
            sum_T2_lag += T[i - m] * T[i - m];

            mu[i - m + 1] = (sum_T - sum_T_lag) / m;
            sigma[i - m + 1] = std::sqrt((sum_T2 - sum_T2_lag) / m - mu[i - m + 1] * mu[i - m + 1]);
        }
    }
}

void stomp(const double *T, double *P, size_t n, size_t m)
{
    ssize_t excl_zone = std::ceil(m / 4.0);

    std::vector<double> QT(n - m + 1), QT2(n - m + 1),QT_first(n - m + 1), D(n - m + 1);
    std::vector<double> mu(n - m + 1), sigma(n - m + 1);

    compute_mean_std(T, mu.data(), sigma.data(), n, m);

    sliding_window_dot_prodouct(T, T, QT.data(), n, m);

    #pragma omp simd
    for (size_t j = 0; j < n - m + 1; j++) {
        QT_first[j] = QT[j];
    }

    #pragma omp simd
    for (size_t j = 0;  j < n - m + 1; j++) {
        P[j] = std::sqrt(2.0 * m * (1.0 - (QT[j] - m * mu[0] * mu[j]) / (m * sigma[0] * sigma[j])));
    }

    for (size_t j = 0; j < excl_zone; j++){
        P[j] = std::numeric_limits<double>::infinity();
    }

    for (size_t j = 0; j < n - m + 1; j++) {
        P[0] = std::min(P[0], P[j]);
    }

    for (size_t i = 1; i < n - m + 1; i++) {
        // Calculate sliding-window dot product
        QT2[0] = QT_first[i];
        #pragma omp simd
        for (size_t j = i; j < n - m + 1; j++) {
            QT2[j] = QT[j - 1] - T[j - 1] * T[i - 1] + T[j + m - 1] * T[i + m - 1];
        }

        // Apply exclusion zone
        for (size_t j = i; j < std::min(i + excl_zone, n - m + 1); j++){
            D[j] = std::numeric_limits<double>::infinity();
        }

        // Calculate distance profile
        #pragma omp simd
        for (size_t j = i + excl_zone;  j < n - m + 1; j++) {
            D[j] = std::sqrt(2.0 * m * (1.0 - (QT2[j] - m * mu[i] * mu[j]) / (m * sigma[i] * sigma[j])));
        }

        // Update matrix profile
        #pragma omp simd
        for (size_t j = i; j < n - m + 1; j++) {
            P[j] = std::min(P[j], D[j]);
        }

         double min_pi = D[i];
         #pragma omp simd reduction(min:min_pi)
         for (size_t j = i + 1; j < n - m + 1; j++) {
             min_pi = std::min(P[i], D[j]);
         }
         P[i] = min_pi;

        #pragma omp simd
        for (size_t j = i; j < n - m + 1; j++) {
            QT[j] = QT2[j];
        }
    }
}
