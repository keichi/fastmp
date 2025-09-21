#include <cmath>

// sliding_window_dot_prodouct()
// {
//     n = T.shape[0]
//     m = Q.shape[0]
// 
//     Ta = np.zeros(2 * n)
//     Ta[:n] = T
//     Qr = Q[::-1]
//     Qra = np.zeros(2 * n)
//     Qra[:m] = Qr
//     Qraf = np.fft.rfft(Qra)
//     Taf = np.fft.rfft(Ta)
//     QT = np.fft.irfft(Qraf * Taf)
//     QT = QT[m-1:n].real
// }

void compute_mean_std(const double *T, double *mu, double *sigma, int n, int m)
{
    double sum_T = 0.0;
    double sum_T_lag = 0.0;
    double sum_T2 = 0.0;
    double sum_T2_lag = 0.0;

    for (int i = 0; i < n; i++) {
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

