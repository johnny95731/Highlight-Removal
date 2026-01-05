import cv2
import numpy as np


def _highlight_distinguish(img: np.ndarray) -> np.ndarray:
    ch_mini = img.min(axis=-1, keepdims=True)
    ch_maxi = img.max(axis=-1, keepdims=True)
    mean, _ = cv2.meanStdDev(ch_mini)
    mean = np.float32(mean[0, 0])

    init_offset = mean
    img_sf = img - ch_mini
    img_msf = img_sf + init_offset

    diff = ch_maxi - ch_mini
    mask1 = (diff > 255 - init_offset)[..., 0]
    img_msf[mask1] = (img_sf + (255 - diff))[mask1]

    mask2 = (ch_maxi < mean)[..., 0]
    img_msf[mask2] = (img_sf + ch_maxi)[mask2]
    return img_msf, ch_maxi, ch_mini


def _count_in_range(gray: np.ndarray, low: float, high: float):
    num = np.count_nonzero(cv2.inRange(gray, low, high))
    return num


def _hist_base_enhance(
    img: np.ndarray,
    img_msf: np.ndarray,
    ch_maxi: np.ndarray,
    ch_mini: np.ndarray,
    thresh1: float,
    thresh2: float,
    thresh3: float,
    thresh4: float,
):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark = _count_in_range(gray, 0, thresh1) + _count_in_range(
        gray, thresh1, thresh2
    )
    bright = _count_in_range(gray, thresh3, thresh4) + _count_in_range(
        gray, thresh4, 255
    )
    if dark > 2 * bright:  # dark type
        msf_maxi = np.max(img_msf, -1, keepdims=True)
        coeff = np.divide(ch_maxi, msf_maxi, out=msf_maxi, where=msf_maxi > 0)
        img_msf *= coeff
    elif bright > 2 * dark:  # bright type
        msf_maxi = np.max(img_msf, -1, keepdims=True)
        coeff = np.divide(ch_mini, msf_maxi, out=msf_maxi, where=msf_maxi > 0)
        img_msf *= coeff
    return img_msf


def _highlight_free(
    img_msf: np.ndarray,
    gamma: float,
    msf_multipler: float,
):
    lum = cv2.cvtColor(img_msf, cv2.COLOR_RGB2GRAY)
    # Highlight free components
    # temp = (1 + cv2.exp(-14 * (lum / 255) ** gamma)) * msf_multipler
    temp = lum / 255
    cv2.pow(temp, gamma, dst=temp)
    cv2.multiply(-14.0, temp, dst=temp)
    cv2.exp(temp, dst=temp)
    np.add(1.0, temp, out=temp)
    highlight_free = temp * msf_multipler
    # Preserve diffuse components
    img_msf *= highlight_free[..., None]
    return img_msf


def saha2020(
    img: np.ndarray,
    thresh1: float = 15,
    thresh2: float = 50,
    thresh3: float = 205,
    thresh4: float = 240,
    gamma_hf: float = 1.5,
    msf_multipler: float = 1.25,
):
    """HDR and highlight removal by Saha's method [1].

    Parameters
    ----------
    img : np.ndarray
        An RGB image in the range of [0, 255] with shape (H, W, 3).
    thresh1 : float, optional
        A threshold for bright checking, by default 15
    thresh2 : float, optional
        A threshold for bright checking, by default 50
    thresh3 : float, optional
        A threshold for dark checking, by default 205
    thresh4 : float, optional
        A threshold for dark checking, by default 240
    gamma_hf : float, optional
        An argument for low-light enhance, by default 1.5
    msf_multipler : float, optional
        An argument for low-light enhance, by default 1.25

    Returns
    -------
    np.ndarray
        HDR image.

    References
    ----------
    [1] P. P. Banik, R. Saha and K. -D. Kim, "HDR image from single LDR
        image after removing highlight," 2018 IEEE International Conference
        on Consumer Electronics (ICCE), Las Vegas, NV, USA, 2018,
        pp. 1-4, https://doi.org/10.1109/ICCE.2018.8326106
    """
    img_msf, ch_maxi, ch_mini = _highlight_distinguish(img)
    img_intr = _hist_base_enhance(
        img,
        img_msf,
        ch_maxi,
        ch_mini,
        thresh1,
        thresh2,
        thresh3,
        thresh4,
    )
    img_hdr = _highlight_free(img_intr, gamma_hf, msf_multipler)
    return img_hdr
