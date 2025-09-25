import numpy as np
import fastmp
import stumpy
import pytest


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_sliding_window_dot_product(n, m):
    T = np.random.rand(n)
    Q = np.random.rand(m)

    QT = fastmp.sliding_window_dot_product(T, Q)

    # Compare to naive calculation
    for i in range(n - m + 1):
        assert np.isclose(T[i:i+m] @ Q, QT[i])

    # Compare to np.convolve
    assert np.allclose(np.convolve(T, Q[::-1], mode="valid"), QT)


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_compute_mean_std(n, m):
    T = np.random.rand(n)

    mu, sigma = fastmp.compute_mean_std(T, m)

    for i in range(n - m + 1):
        assert np.isclose(np.mean(T[i:i+m]), mu[i])
        assert np.isclose(np.std(T[i:i+m]), sigma[i])


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_stomp(n, m):
    T = np.random.rand(n)

    mp = fastmp.stomp(T, m)
    mp2 = stumpy.stump(T, m)[:, 0].astype(np.float64)

    assert np.allclose(mp, mp2)
