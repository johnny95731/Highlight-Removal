import math

import numpy as np
from numba import njit, types


@njit(
    [types.Array(types.float32, 2, 'C')(types.Array(types.float32, 2, 'C'))],
    nogil=True,
    cache=True,
    fastmath=True,
)
def _normalize_axis0(data: np.ndarray):
    for j in range(data.shape[1]):
        norm = 0.0
        for i, val in enumerate(data[:, j]):
            norm += val**2
        norm = math.sqrt(norm)
        for i in range(data.shape[0]):
            data[i, j] /= norm
    return data


@njit(
    [types.Array(types.float32, 1, 'C')(types.Array(types.float32, 2, 'C'))],
    nogil=True,
    cache=True,
    fastmath=True,
)
def _sum_axis0(data: np.ndarray):
    res = np.zeros(data.shape[1], dtype=data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            res[j] += data[i, j]
    return res


@njit(
    [
        types.Tuple((
            types.Array(types.float32, 2, 'C'),
            types.Array(types.float32, 2, 'C'),
            types.float64,
        ))(
            types.Array(types.float32, 2, 'C'),
            types.Array(types.float32, 2, 'C'),
            types.Array(types.float32, 2, 'C'),
            types.float32,
            types.Array(types.float32, 2, 'C'),
        )
    ],
    nogil=True,
    cache=True,
    fastmath=True,
    error_model='numpy',
)
def _update_mats(
    data,  # shape (ch, N)
    mat_w,  # shape (ch, R)
    mat_h,  # shape (R, N)
    _lambda,  # float
    i_s,  # shape (ch, 1)
):
    _normalize_axis0(mat_w)
    # Update mat_h
    w_bar = np.ascontiguousarray(mat_w.transpose(1, 0))  # shape (R, ch)
    temp1 = w_bar @ data  # shape (R, N)
    temp2 = w_bar @ mat_w @ mat_h  # shape (R, N)
    for i, row in enumerate(temp1):
        for j, val in enumerate(row):
            mat_h[i, j] *= val / (temp2[i, j] + _lambda)

    Vl = data - (i_s @ np.ascontiguousarray(mat_h[:1, :]))
    for i, row in enumerate(Vl):
        for j, val in enumerate(row):
            if val < 0:
                row[j] = 0

    H_d = np.ascontiguousarray(mat_h[1:, :])
    H_d_t = np.ascontiguousarray(H_d.transpose(1, 0))
    mat_w_d = np.ascontiguousarray(mat_w[:, 1:])

    # Update mat_w
    temp_mat = mat_w_d @ H_d @ H_d_t
    temp_Vl = Vl @ H_d_t  # shape (ch, R-1)
    temp3 = _sum_axis0(temp_mat)
    temp4 = _sum_axis0(temp_Vl)
    for i, row in enumerate(temp_Vl):
        for j, val in enumerate(row):
            mat_w_d[i, j] *= (val + mat_w_d[i, j] * temp3[j]) / (
                temp_mat[i, j] + mat_w_d[i, j] * temp4[j]
            )
    mat_w[:, 0] = i_s[:, 0]
    mat_w[:, 1:] = mat_w_d

    d = data - mat_w @ mat_h
    F_t = 0.5 * np.linalg.norm(d) + _lambda * np.sum(mat_h)
    return mat_w, mat_h, F_t


@njit(
    [
        types.Array(types.float32, 2, 'C')(
            types.Array(types.float32, 2, 'C'),
            types.int64,
            types.float64,
            types.float64,
            types.int64,
        ),
    ],
    nogil=True,
    cache=True,
    fastmath=True,
    error_model='numpy',
)
def _akashi2016(
    data: np.ndarray,
    R: int,
    _lambda: float,
    rtol: float,
    max_iter: int,
):
    channels, num_pixels = data.shape
    _lambda = np.float32(_lambda)

    mat_i_s = np.full((channels, 1), np.sqrt(1 / 3), dtype=np.float32)
    mat_w_d = np.random.uniform(1, 255, size=(channels, R - 1)).astype(
        np.float32
    )
    _normalize_axis0(mat_w_d)

    mat_w = np.empty((channels, R), dtype=np.float32)
    mat_w[:, 0] = mat_i_s[:, 0]
    mat_w[:, 1:] = mat_w_d

    mat_h = np.random.uniform(1, 255, size=(R, num_pixels)).astype(np.float32)

    F_t_1 = np.inf
    for _ in range(max_iter):
        mat_w, mat_h, F_t = _update_mats(data, mat_w, mat_h, _lambda, mat_i_s)
        if abs(F_t - F_t_1) < rtol * abs(F_t):
            break
        F_t_1 = F_t

    res = np.ascontiguousarray(mat_w[:, 1:]) @ np.ascontiguousarray(
        mat_h[1:, :]
    )
    return res


def akashi2016(
    img: np.ndarray,
    R: int = 15,
    _lambda: float = 3.0,
    rtol: float = 1e-8,
    max_iter: int = 100,
):
    height, width, channels = img.shape
    num_pixels = height * width

    # shape = (channels, num_pixels)
    flatted = img.reshape(num_pixels, channels).transpose(1, 0)
    flatted = np.ascontiguousarray(flatted, np.float32)

    res = _akashi2016(flatted, R, _lambda, rtol, max_iter)

    res = res.transpose(1, 0).reshape(height, width, channels)
    res = np.ascontiguousarray(res)
    res = np.clip(res, 0, 255)
    return res
