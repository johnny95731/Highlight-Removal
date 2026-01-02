import cv2
import numpy as np


def _detect_specular(img: np.ndarray):
    ch_maxi = img.max(-1)  # types: np.ndarray
    ch_mini = img.min(-1)  # types: np.ndarray

    coeff = np.divide(
        ch_mini,
        ch_maxi,
        out=np.zeros_like(ch_maxi, dtype=np.float32),
        where=ch_maxi > 0,
    )

    enhanced = coeff[..., None] * img
    xyz = cv2.cvtColor(enhanced, cv2.COLOR_RGB2XYZ)

    # xyz = xyz / 255
    # lum = xyz[..., 1]
    # chromatic_lum = xyz[..., 1] / np.sum(xyz, axis=-1)
    # mask_specular = chromatic_lum <= lum
    mask_specular = np.sum(xyz, axis=-1, out=ch_maxi) >= 255  # Simpler
    return mask_specular


def meslouhi2011(img: np.ndarray, n: int = 16, radius: float = 5):
    """An simpler approximate of El Meslouhi's highlight removal [1]. (or,
    [9] in README.md)

    Parameters
    ----------
    img : np.ndarray
        An RGB image in the range of [0, 255] with shape (H, W, 3).
    n : int, optional
        Minimum restored area. Ignore the regions which area less than `n**2`
        By default 16.
    radius : float, optional
        Inpaint radius, by default 50

    Returns
    -------
    np.ndarray
        Image without highlight. shape=input. dtype=uint8.
    """
    mask_specular = _detect_specular(np.float32(img))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_specular.astype(np.uint8), connectivity=8
    )

    min_area = n**2
    temp = np.uint8(img)
    # keys = []
    for i in range(1, num_labels):
        block_area = stats[i, 2] * stats[i, 3]
        if block_area < min_area:
            continue
        # Not the required inpaint algorithm.
        temp = cv2.inpaint(
            temp,
            (labels == i).astype(np.uint8),
            radius,
            cv2.INPAINT_NS,
        )
    #     keys.append(i)
    # # step 3 (histogram equalization). The result is not good.
    # lab = cv2.cvtColor(temp, cv2.COLOR_RGB2LAB)
    # for i in keys:
    #     left = stats[i, 0]
    #     top = stats[i, 1]
    #     right = left + stats[i, 2]
    #     bottom = top + stats[i, 3]
    #     region = lab[top:bottom, left:right, 0]
    #     mask = labels[top:bottom, left:right] == i
    #     region[mask] = cv2.equalizeHist(region)[mask]
    # res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return temp
