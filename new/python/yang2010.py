import sys

import cv2
import numpy as np


def yang2010(
    img: np.ndarray,
    sigma_color: float = 0.04,
    sigma_space: float = 0.25,
    tol: float = 0.03,
    max_iter: int = 10,
):
    """Highlight removal by Yang's method [1]. (or, [6] in README.md)

    Parameters
    ----------
    img : np.ndarray
        An RGB image in the range of [0, 255] with shape (H, W, 3).
    sigma_color : float, optional
        An argument for bilateral filter, by default 0.04
    sigma_space : float, optional
        An argument for bilateral filter, by default 0.25
    tol : float, optional
        Condition for stop iteration, by default 0.03
    max_iter : int, optional
        Maximum iteration number, by default 10

    Returns
    -------
    np.ndarray
        Image without highlight. shape=input. dtype=float32.

    References
    ----------
    [1] Yang, Q., Wang, S., Ahuja, N. "Real-Time Specular Highlight Removal
    Using Bilateral Filtering," Computer Vision - ECCV 2010, vol 6314,
    pp. 87-100, https://doi.org/10.1007/978-3-642-15561-1_7
    """
    if img.ndim < 3:
        raise ValueError(
            f'`img` mush be an RGB image: img.ndim = {img.ndim} < 3'
        )
    elif img.shape[-1] != 3:
        raise ValueError(
            f'`img` mush be an RGB image: channels = {img.shape[-1]} != 3'
        )
    img = np.divide(img, 255, dtype=np.float32)
    _sum = np.sum(img, axis=-1, keepdims=True)

    sigma = np.divide(img, _sum, out=np.zeros_like(img), where=_sum > 0)

    sigma_max = np.max(sigma, axis=-1)
    for _ in range(max_iter):
        temp = cv2.bilateralFilter(sigma_max, -1, sigma_color, sigma_space)
        if np.count_nonzero(temp - sigma_max > tol) == 0:
            break
        sigma_max = np.maximum(sigma_max, temp)

    ch_maxi = np.max(img, axis=-1, keepdims=True)

    sigma_max = np.expand_dims(sigma_max, 2)
    den = 1 - 3 * sigma_max
    mask_nonzero = den != 0
    img_sp = np.divide(
        ch_maxi - sigma_max * _sum,
        den,
        out=np.zeros_like(ch_maxi),
        where=mask_nonzero,
    )
    mask_zero = np.bitwise_not(mask_nonzero)
    img_sp[mask_zero] = np.max(img_sp[mask_nonzero])

    res = np.clip(img - img_sp, 0.0, 1.0)
    res = cv2.convertScaleAbs(res, None, 255)
    return res


img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR_RGB)
res = yang2010(img)
cv2.imwrite('./yang2010.jpeg', res[:, :, ::-1])
cv2.imwrite('./yang2010_ori.jpeg', img[:, :, ::-1])
