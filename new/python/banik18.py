import cv2
import numpy as np


def _highlight_distinguish(
    img: np.ndarray,
    eta_msf: float,
) -> np.ndarray[tuple[int, int], np.bool_]:
    ch_mini = img.min(axis=-1)
    mean, std = cv2.meanStdDev(ch_mini)
    mean = float(mean[0, 0])
    std = float(std[0, 0])

    thresh_offset = mean + eta_msf * std
    thresh_hl = 2 * mean
    mask_highlight = ch_mini > min(thresh_offset, thresh_hl)

    temp = np.where(ch_mini > thresh_offset, thresh_offset - ch_mini, 0.0)
    img_msf = img + np.expand_dims(temp, 2)
    return mask_highlight, img_msf


def _highlight_free(
    img_msf: np.ndarray, mask_highlight: np.ndarray, msf_multipler: float
):
    lum = cv2.cvtColor(img_msf, cv2.COLOR_RGB2GRAY)
    # Highlight free components
    hl = lum[mask_highlight]
    # temp = 1 + cv2.exp(-14 * (lum / 255) ** 1.6)
    temp = hl / 255
    cv2.pow(temp, 1.6, dst=temp)
    cv2.multiply(-14.0, temp, dst=temp)
    cv2.exp(temp, dst=temp)
    np.add(1.0, temp, out=temp)
    highlight_free = temp * msf_multipler
    # Preserve diffuse components
    img_msf[mask_highlight] *= highlight_free[:, None]
    return img_msf


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
    mask_highlight, img_msf = _highlight_distinguish(img, eta_msf)
    img_hf = _highlight_free(img_msf, mask_highlight, msf_multipler)
    lum_hf = cv2.cvtColor(img_hf, cv2.COLOR_RGB2GRAY)

    lum_hf_log = np.add(lum_hf, eps)
    cv2.log(lum_hf_log, dst=lum_hf_log)
    lum_hf_log_mean = cv2.mean(lum_hf_log)
    lum_hf_exp_log_mean = cv2.exp(lum_hf_log_mean)

    scaled_lum_hf = cv2.multiply(
        lum_hf, lum_scaler / lum_hf_exp_log_mean, dtype=cv2.CV_32F
    )

    ones = np.ones_like(scaled_lum_hf)
    # tone mapped
    # scaled_lum_hf * (1 + (scaled_lum_hf / 0.35**2)) / (1 + scaled_lum_hf)
    temp_a = cv2.scaleAdd(scaled_lum_hf, lum_white ** (-2), ones)
    temp_b = cv2.add(ones, scaled_lum_hf, dst=ones)
    temp_a /= temp_b
    tone_lum_hf = cv2.multiply(scaled_lum_hf, temp_a, dst=scaled_lum_hf)

    # img_hf * lum_hf / (255 * tone_lum_hf)
    cv2.divide(lum_hf, tone_lum_hf, scale=1 / 255, dst=lum_hf, dtype=cv2.CV_32F)
    lum_hf = np.expand_dims(lum_hf, 2)
    img_hdr = np.multiply(img_hf, lum_hf, out=img_hf)
    return img_hdr
