import math

import numpy as np
from numba import njit, types


@njit(
    [
        types.float32(
            types.Array(types.float32, 1, 'C', readonly=True),
            types.Array(types.float32, 1, 'C', readonly=True),
        )
    ],
    cache=True,
    nogil=True,
)
def _dot(a, b):
    res = np.float32(0.0)
    for i in range(3):
        res += a[i] * b[i]
    return res


@njit(
    [
        types.Tuple((
            types.Array(types.float32, 3, 'C'),
            types.Array(types.float32, 2, 'C'),
        ))(
            types.Array(types.float32, 3, 'C', readonly=True),
            types.float32,
            types.float32,
        )
    ],
    cache=True,
    nogil=True,
)
def _hl_free_image(img: np.ndarray, eta_msf: float, msf_multipler: float):
    # highlight distinguish
    ch_mini = np.empty(img.shape[:2], img.dtype)
    mean = 0.0
    sq_mean = 0.0
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            mini = min(pixel)
            ch_mini[y, x] = mini
            mean += mini
            sq_mean += mini**2
    mean /= img.shape[0] * img.shape[1]
    sq_mean /= img.shape[0] * img.shape[1]
    std = sq_mean - mean**2

    thresh_offset = np.float32(mean + eta_msf * std)
    thresh = min(thresh_offset, 2 * mean)
    # highlight free
    img_hf = np.empty_like(img)
    lum_hf = np.empty(img.shape[:2], dtype=img.dtype)
    weight = np.array((0.299, 0.587, 0.114), dtype=np.float32)
    for y, row in enumerate(ch_mini):
        for x, mini in enumerate(row):
            if mini > thresh:
                msf = img[y, x] + (thresh_offset - mini)
                lum = _dot(msf, weight)
                # Suppress highlight components
                temp = -14.0 * (lum / 255) ** 1.6
                img_hf[y, x] = img[y, x] * (
                    (1.0 + math.exp(temp)) * msf_multipler
                )
            else:
                # preserve diffuse components
                img_hf[y, x] = img[y, x]
            lum_hf[y, x] = _dot(img_hf[y, x], weight)
    return img_hf, lum_hf


@njit(
    [
        types.float32(
            types.Array(types.float32, 2, 'C', readonly=True),
            types.float32,
        )
    ],
    cache=True,
    nogil=True,
)
def _geometric_mean(lum: np.ndarray, eps: float):
    sum_log = 0.0
    for row in lum:
        for val in row:
            sum_log += math.log(eps + val)
    mean_log = sum_log / lum.size
    geo_mean = math.exp(mean_log)
    return geo_mean


@njit(
    [
        types.Array(types.float32, 2, 'C')(
            types.Array(types.float32, 2, 'C'),
            types.float32,
            types.float32,
            types.float32,
        )
    ],
    cache=True,
    nogil=True,
)
def _tone_multipler(
    lum: np.ndarray,
    eps: float,
    lum_scaler: float,
    lum_white: float,
):
    geo_mean = _geometric_mean(lum, eps)
    coeff = lum_scaler / geo_mean

    coeff2 = lum_white**2
    for y, row in enumerate(lum):
        for x, val in enumerate(row):
            scaled = val * coeff
            tone = scaled * ((coeff2 + scaled) / (coeff2 + coeff2 * scaled))
            lum[y, x] /= 255 * tone
    return lum


@njit(
    [
        types.Array(types.float32, 3, 'C')(
            types.Array(types.float32, 3, 'C'),
            types.Array(types.float32, 2, 'C', readonly=True),
        )
    ],
    cache=True,
    nogil=True,
)
def _hdr(img_hf: np.ndarray, tone_lum_hf: np.ndarray):
    for y, row in enumerate(tone_lum_hf):
        for x, val in enumerate(row):
            for c in range(3):
                img_hf[y, x, c] *= val
    return img_hf


def banik18(
    img: np.ndarray,
    eta_msf: float = 2.5,
    msf_multipler: float = 1.025,
    eps: float = 1e-10,
    lum_scaler: float = 0.05,
    lum_white: float = 0.35,
):
    """Highlight removal by Banik's method [1]. (or, [15] in README.md)

    Parameters
    ----------
    img : np.ndarray
        An RGB image in the range of [0, 255] with shape (H, W, 3).
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
    np.ndarray
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
    elif img.shape[-1] != 3:
        raise ValueError(
            f'`img` mush be an RGB image: channels = {img.shape[-1]} != 3'
        )
    img = np.float32(img)
    img_hf, lum_hf = _hl_free_image(img, eta_msf, msf_multipler)

    tone_lum_hf = _tone_multipler(lum_hf, eps, lum_scaler, lum_white)
    img_hdr = _hdr(img_hf, tone_lum_hf)
    return img_hdr
