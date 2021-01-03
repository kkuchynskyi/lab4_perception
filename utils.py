from numba import jit
import numpy as np

#
# def call_q(img, i, j, c):
#     """
#     >>> call_q(np.array([[24, 1],[123, 190]]), 0, 0 , 12)
#     -12
#     >>> call_q(np.array([[24, 1],[123, 190]]), 0, 1 , 12)
#     -11
#     >>> call_q(np.array([[24, 1],[123, 190]]), 1, 0 , 12)
#     -111
#     >>> call_q(np.array([[24, 1],[123, 190]]), 1, 1 , 12)
#     -178
#     >>> call_q(np.array([[100, 100],[100, 100]]), 0, 0, 50)
#     -50
#     >>> call_q(np.array([[100, 100],[100, 100]]), 0, 1, 50)
#     -50
#     >>> call_q(np.array([[100, 100],[100, 100]]), 1, 0, 50)
#     -50
#     >>> call_q(np.array([[100, 100],[100, 100]]), 1, 1, 50)
#     -50
#     >>> call_q(np.array([[255, 255],[255, 255]]), 0, 0, 155)
#     -100
#     >>> call_q(np.array([[255, 255],[255, 255]]), 0, 1, 155)
#     -100
#     >>> call_q(np.array([[255, 255],[255, 255]]), 1, 0, 155)
#     -100
#     >>> call_q(np.array([[255, 255],[255, 255]]), 1, 1, 155)
#     -100
#     """
#     if img[i, j] != 0:
#         return -abs(img[i, j] - c)
#     else:
#         return 0


def restore_k(i, j, C, L, R, q, phi):
    values = list()
    for k_ in range(len(C)):
        values.append(L[i, j, k_] + R[i, j, k_] + q[i, j, k_] - phi[i, j, k_])
    return C[values.index(max(values))]


@jit(nopython=True)
def update_left(i, j, k, direction, phi, C, g, q):
    values = np.zeros((len(C),), dtype=np.float32)
    for k_ in range(len(C)):
        values[k_] = direction[i, j-1, k_] + 0.5*q[i, j-1, k_] - phi[i, j-1, k_] + g[i, j, 2, k_, k]
    return values.max()


@jit(nopython=True)
def update_upper(i, j, k, direction, phi, C, g, q):
    values = np.zeros((len(C),), dtype=np.float32)
    for k_ in range(len(C)):
        values[k_] = direction[i-1, j, k_] + 0.5*q[i-1, j, k_] + phi[i-1, j, k_] + g[i, j, 0, k_, k]
    return values.max()


@jit(nopython=True)
def update_right(i, j, k, direction, phi, C, g, q):
    values = np.zeros((len(C),), dtype=np.float32)
    for k_ in range(len(C)):
        values[k_] = direction[i, j+1, k_] + 0.5*q[i, j+1, k_] - phi[i, j+1, k_] + g[i, j, 3, k_, k]
    return values.max()


@jit(nopython=True)
def update_down(i, j, k, direction, phi, C, g, q):
    values = np.zeros((len(C),), dtype=np.float32)
    for k_ in range(len(C)):
        values[k_] = direction[i+1, j, k_] + 0.5*q[i+1, j, k_] + phi[i+1, j, k_] + g[i, j, 1, k_, k]
    return values.max()


@jit(nopython=True)
def trws(m, n, C, g, q):
    L = np.zeros((m, n, len(C)), dtype=np.float64)
    U = np.zeros((m, n, len(C)), dtype=np.float64)
    D = np.zeros((m, n, len(C)), dtype=np.float64)
    R = np.zeros((m, n, len(C)), dtype=np.float64)
    phi = np.zeros((m, n, len(C)))

    for i in range(m - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            for c in range(len(C)):
                R[i, j, c] = update_right(i, j, c, R, phi, C, g, q)
                D[i, j, c] = update_down(i, j, c, D, phi, C, g, q)

    for iteration in range(100):
        # forward
        for i in range(1, m):
            for j in range(1, n):
                for c in range(len(C)):
                    L[i, j, c] = update_left(i, j, c, L, phi, C, g, q)
                    U[i, j, c] = update_upper(i, j, c, U, phi, C, g, q)
                    phi[i, j, c] = (L[i, j, c] - U[i, j, c] + R[i, j, c] - D[i, j, c]) / 2
        # backward
        for i in range(m - 2, -1, -1):
            for j in range(n - 2, -1, -1):
                for c in range(len(C)):
                    R[i, j, c] = update_right(i, j, c, R, phi, C, g, q)
                    D[i, j, c] = update_down(i, j, c, D, phi, C, g, q)
                    phi[i, j, c] = (L[i, j, c] - U[i, j, c] + R[i, j, c] - D[i, j, c]) / 2
    return L, U, D, R, phi