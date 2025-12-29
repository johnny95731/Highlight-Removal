import torch


def _rgb2gray(img: torch.Tensor):
    weights = torch.tensor(
        (0.299, 0.587, 0.114), dtype=torch.float32, device=img.device
    )
    img = img.movedim(-3, -1)
    res = torch.matmul(img, weights)
    return res


def _highlight_distinguish(
    img: torch.Tensor,
    eta_msf: float,
) -> torch.Tensor:
    ch_mini = img.amin(-3)
    mean = ch_mini.mean()
    std = ch_mini.std()

    thresh_offset = mean + eta_msf * std
    thresh_hl = 2 * mean
    mask_highlight = ch_mini > min(thresh_offset, thresh_hl)

    temp = torch.where(ch_mini > thresh_offset, thresh_offset - ch_mini, 0.0)
    img_msf = img + temp.unsqueeze_(-3)
    return mask_highlight, img_msf


def _highlight_free(
    img_msf: torch.Tensor,
    mask_highlight: torch.Tensor,
    msf_multipler: float,
):
    lum = _rgb2gray(img_msf)
    # Highlight free components
    hl = lum[mask_highlight]
    temp = 1.0 + torch.exp(-14.0 * (hl / 255.0).pow(1.6))
    lum[mask_highlight] = temp * msf_multipler
    lum[mask_highlight.bitwise_not_()] = 1.0
    # Preserve diffuse components
    img_msf *= lum.unsqueeze_(-3)
    return img_msf


def banik18(
    img: torch.Tensor,
    eta_msf: float = 2.5,
    msf_multipler: float = 1.025,
    eps: float = 1e-10,
    lum_scaler: float = 0.05,
    lum_white: float = 0.35,
):
    """Highlight removal by Banik's method [1]. (or, [15] in README.md)

    Parameters
    ----------
    img : torch.Tensor
        An RGB image in the range of [0, 255] with shape (*, 3, H, W).
    eta_msf : float, optional
        An coefficient for detecting highlight area. Higher value means
        'strict'. By default 2.5.
    msf_multipler : float, optional
        A scaler of modified specular free image. By default 1.025.
    eps : float, optional
        An small bias when calculating the geometric mean, by default 1e-10.
    lum_scaler : float, optional
        Luminance scaler, by default 0.05.
    lum_white : float, optional
        An coefficient for tone mapped luminance, by default 0.35.

    Returns
    -------
    torch.Tensor
        Image without highlight. shape=input. dtype=float32.

    References
    ----------
    [1] P. P. Banik, R. Saha and K. -D. Kim, "HDR image from single LDR image
    after removing highlight," 2018 IEEE International Conference on
    Consumer Electronics (ICCE), Las Vegas, NV, USA, 2018, pp. 1-4,
    https://doi.org/10.1109/ICCE.2018.8326106
    """
    if img.ndim < 3:
        raise ValueError(
            f'`img` mush be an RGB image: img.ndim = {img.ndim} < 3'
        )
    elif img.size(-3) != 3:
        raise ValueError(
            f'`img` mush be an RGB image: channels = {img.size(-3)} != 3'
        )
    elif not torch.is_floating_point(img):
        img = img.float()

    is_not_batch = img.ndim == 3
    if is_not_batch:
        img.unsqueeze_(0)

    mask_highlight, img_msf = _highlight_distinguish(img, eta_msf)
    img_hf = _highlight_free(img_msf, mask_highlight, msf_multipler)
    lum_hf = _rgb2gray(img_hf)

    lum_hf_exp_log_mean = (eps + lum_hf).log().mean().exp()
    scaled_lum_hf = lum_hf * (lum_scaler / lum_hf_exp_log_mean)

    # tone mapped
    # scaled_lum_hf * (1 + (scaled_lum_hf / 0.35**2)) / (1 + scaled_lum_hf)
    temp_a = 1.0 + scaled_lum_hf / lum_white ** (2)
    temp_b = 1.0 + scaled_lum_hf
    temp_a /= temp_b
    tone_lum_hf = scaled_lum_hf * temp_a

    # img_hf * lum_hf / (255 * tone_lum_hf)
    temp = (lum_hf / (255.0 * tone_lum_hf)).unsqueeze_(-3)
    img_hdr = img_hf * temp
    if is_not_batch:
        img_hdr.squeeze_(0)
    return img_hdr
