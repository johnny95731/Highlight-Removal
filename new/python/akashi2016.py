import numpy as np


def separate_ds(
    data,  # shape (ch, N)
    mat_w,  # shape (ch, R)
    mat_h,  # shape (R, N)
    _lambda,  # float
    i_s,  # shape (ch, 1)
):
    mat_w /= np.linalg.norm(mat_w, 2, axis=0)

    w_bar = mat_w.transpose(1, 0)
    mat_h = mat_h * (w_bar @ data) / (w_bar @ mat_w @ mat_h + _lambda)

    Vl = data - (i_s @ mat_h[:1, :])
    np.clip(Vl, 0.0, None, out=Vl)

    mat_h_d = mat_h[1:, :]
    mat_h_d_t = mat_h_d.transpose(1, 0)
    mat_w_d_bar = mat_w[:, 1:]

    temp_mat = mat_w_d_bar @ mat_h_d @ mat_h_d_t
    temp_Vl = Vl @ mat_h_d_t
    mat_w_d_bar = (
        mat_w_d_bar
        * (temp_Vl + mat_w_d_bar * np.sum(temp_mat, axis=0))
        / (temp_mat + mat_w_d_bar * np.sum(temp_Vl, axis=0))
    )

    mat_w[:, 0] = i_s[:, 0]
    mat_w[:, 1:] = mat_w_d_bar

    d = data - mat_w @ mat_h
    F_t = 0.5 * np.linalg.norm(d, 'fro') + _lambda * np.sum(mat_h)

    return mat_w, mat_h, F_t


def akashi2016(
    img: np.ndarray,
    R: int = 7,
    _lambda: float = 3.0,
    rtol: float = 1e-10,
    max_iter: int = 100,
):
    """Image highlight removal. An implementation of
    Akashi et al. (see [12] in README.md).

    Parameters
    ----------
    img : np.ndarray
        An image with whape (H, W, C).
    R : int, optional
        The number of diffuse colors plus one, by default 7.
    _lambda : float, optional
        An coefficient of loss function, by default 3.0.
    rtol : float, optional
        Relative tolerance for stopping iteration, by default 1e-10.
    max_iter : int, optional
        Maximum number of iterations, by default 100.

    Returns
    -------
    np.ndarray
        The diffuse components.
    """
    img = np.asarray(img)
    height, width, channels = img.shape
    num_pixels = height * width

    # shape = (channels, num_pixels)
    flatted = img.reshape(num_pixels, channels).transpose(1, 0)
    flatted = np.ascontiguousarray(flatted, np.float32)

    mat_i_s = np.full((channels, 1), np.sqrt(1 / 3), dtype=np.float32)
    mat_w_d = np.random.uniform(1, 255, size=(channels, R - 1)).astype(
        np.float32
    )
    mat_w_d /= np.linalg.norm(mat_w_d, 2, axis=0)

    mat_w = np.concat((mat_i_s, mat_w_d), axis=1, dtype=np.float32)
    mat_h = np.random.uniform(1, 255, size=(R, num_pixels)).astype(np.float32)

    F_t_1 = np.inf
    for _ in range(max_iter):
        mat_w, mat_h, F_t = separate_ds(flatted, mat_w, mat_h, _lambda, mat_i_s)
        if abs(F_t - F_t_1) < rtol * abs(F_t):
            break
        F_t_1 = F_t

    res = mat_w[:, 1:] @ mat_h[1:, :]

    res = res.transpose(1, 0).reshape(height, width, channels)
    res = np.ascontiguousarray(res)
    res = np.clip(res, 0, 255)
    return res
