import torch


def separate_ds(
    data: torch.Tensor,  # shape (ch, N)
    mat_w: torch.Tensor,  # shape (ch, R)
    mat_h: torch.Tensor,  # shape (R, N)
    _lambda: torch.Tensor,  # float
    i_s: torch.Tensor,  # shape (ch, 1)
):
    mat_w /= torch.linalg.norm(mat_w, axis=0)

    w_bar = mat_w.transpose(1, 0)
    mat_h *= (w_bar @ data) / (w_bar @ mat_w @ mat_h + _lambda)

    Vl = data - (i_s @ mat_h[:1, :])
    torch.clip(Vl, 0.0, None, out=Vl)

    mat_h_d = mat_h[1:, :]
    mat_h_d_t = mat_h_d.transpose(1, 0)
    mat_w_d_bar = mat_w[:, 1:]

    temp_mat = mat_w_d_bar @ mat_h_d @ mat_h_d_t
    temp_Vl = Vl @ mat_h_d_t
    mat_w_d_bar *= (temp_Vl + mat_w_d_bar * torch.sum(temp_mat, axis=0)) / (
        temp_mat + mat_w_d_bar * torch.sum(temp_Vl, axis=0)
    )

    mat_w[:, 0] = i_s[:, 0]
    mat_w[:, 1:] = mat_w_d_bar

    d = data - mat_w @ mat_h
    F_t = 0.5 * torch.linalg.norm(d) + _lambda * torch.sum(mat_h)

    return mat_w, mat_h, F_t


def akashi2016(
    img: torch.Tensor,
    R: int = 7,
    _lambda: float = 3.0,
    rtol: float = 1e-8,
    max_iter: int = 100,
):
    """Image highlight removal. An implementation of
    Akashi et al. (see [12] in README.md).

    Parameters
    ----------
    img : torch.Tensor
        An image with whape (C, H, W).
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
    torch.Tensor
        The diffuse components.
    """
    img = torch.as_tensor(img)
    channels, height, width = img.shape
    num_pixels = height * width

    # shape = (channels, num_pixels)
    flatted = img.reshape(channels, num_pixels).contiguous()
    flatted = flatted.type(torch.float32)

    mat_i_s = torch.full((channels, 1), (1 / 3) ** 0.5, dtype=torch.float32)
    mat_w_d = torch.rand(size=(channels, R - 1), dtype=torch.float32)
    mat_w_d /= torch.linalg.norm(mat_w_d, dim=0)

    mat_w = torch.concat((mat_i_s, mat_w_d), dim=1)
    mat_h = torch.rand(size=(R, num_pixels), dtype=torch.float32)

    F_t_1 = torch.inf
    for _ in range(max_iter):
        mat_w, mat_h, F_t = separate_ds(flatted, mat_w, mat_h, _lambda, mat_i_s)
        if torch.abs(F_t - F_t_1) < rtol * torch.abs(F_t):
            break
        F_t_1 = F_t

    res = mat_w[:, 1:] @ mat_h[1:, :]

    res = res.reshape(channels, height, width).contiguous()
    torch.clip_(res, 0, 255)
    return res
