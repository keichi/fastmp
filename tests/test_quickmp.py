import numpy as np
import quickmp
import stumpy
import pytest


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_sliding_window_dot_product(n, m):
    T = np.random.rand(n)
    Q = np.random.rand(m)

    QT = quickmp.sliding_window_dot_product(T, Q)

    # Compare to naive calculation
    for i in range(n - m + 1):
        assert np.isclose(T[i:i+m] @ Q, QT[i])

    # Compare to np.convolve
    assert np.allclose(np.convolve(T, Q[::-1], mode="valid"), QT)


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_compute_mean_std(n, m):
    T = np.random.rand(n)

    mu, sigma = quickmp.compute_mean_std(T, m)

    for i in range(n - m + 1):
        assert np.isclose(np.mean(T[i:i+m]), mu[i])
        assert np.isclose(np.std(T[i:i+m]), sigma[i])


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_selfjoin(n, m):
    T = np.random.rand(n)

    mp = quickmp.selfjoin(T, m)
    mp2 = stumpy.stump(T, m)[:, 0].astype(np.float64)

    assert np.allclose(mp, mp2)

@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_abjoin(n, m):
    T1 = np.random.rand(n)
    T2 = np.random.rand(n)

    mp = quickmp.abjoin(T1, T2, m)

    mp2 = stumpy.stump(T_A=T1, T_B=T2, m=m, ignore_trivial=False)[:, 0].astype(np.float64)
    assert np.allclose(mp, mp2)
