import numpy as np
from numba import njit, types


@njit(
    [
        types.UniTuple(types.int64, 2)(
            types.Array(types.float32, 2, 'C', readonly=True),
            types.Array(types.boolean, 2, 'C', readonly=True),
            types.Array(types.boolean, 2, 'C', readonly=True),
        )
    ],
    nogil=True,
    cache=True,
)
def _masked_argmax(img: np.ndarray, mask1: np.ndarray, mask2: np.ndarray):
    maxi = float('-inf')
    index = (-1, -1)
    for y, row in enumerate(img):
        for x, val in enumerate(row):
            if mask1[y, x] and mask2[y, x] and val > maxi:
                index = (y, x)
    return index


@njit(
    [
        types.float32(
            types.Array(types.float32, 1, 'A', readonly=True),
            types.Array(types.float32, 1, 'C', readonly=True),
        )
    ],
    nogil=True,
    cache=True,
)
def _dot1d(a: np.ndarray, b: np.ndarray):
    res = types.float32(0.0)
    for i, val in enumerate(a):
        res += val * b[i]
    return res


@njit(
    [
        types.Array(types.float32, 2, 'C')(
            types.Array(types.float32, 3, 'C', readonly=True),
        )
    ],
    nogil=True,
    cache=True,
)
def _ch_abs_sum(img: np.ndarray):
    _sum = np.empty(img.shape[:2], np.float32)
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            s = 0.0
            for val in pixel:
                s += abs(val)
            _sum[y, x] = s
    return _sum


@njit(
    [
        types.void(
            types.Array(types.float32, 3, 'C'),
            types.Array(types.float32, 3, 'C'),
            types.Array(types.float32, 3, 'C'),
            types.Array(types.float32, 2, 'C'),
            types.Array(types.float32, 1, 'C', readonly=True),  # illum
            types.Array(types.float32, 3, 'C'),
            types.float32,  # threshold_chroma
            types.Array(types.boolean, 2, 'C'),
            types.Array(types.boolean, 2, 'C'),
        )
    ],
    nogil=True,
    cache=True,
)
def _handle_diffuse(
    img: np.ndarray,
    img_df: np.ndarray,
    img_coeff: np.ndarray,
    ch_maxi: np.ndarray,
    illum: np.ndarray,
    chroma: np.ndarray,
    threshold_chroma: float,
    unprocessed: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    mask_diffuse: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
):
    h, w, _ = img.shape
    while True:
        y, x = _masked_argmax(ch_maxi, unprocessed, mask_diffuse)
        if y == -1 or x == -1:
            break
        # Regard the pixel as body color
        color_body = img_df[y, x]
        chroma_body = chroma[y, x]
        vcomb = np.vstack((color_body, illum)).T

        # Chromaticity difference
        # Exclude non-diffuse and non-combined reflection pixels
        c_diff_sum = _ch_abs_sum(chroma - chroma_body)

        # Let the pixel be the diffuse component, then solve the reflection coefficient
        vcomb_pinv = np.linalg.pinv(vcomb)
        vb_pinv = np.linalg.pinv(color_body[:, np.newaxis])  # shape (3)
        for y0 in range(h):
            for x0 in range(w):
                if (
                    c_diff_sum[y0, x0] >= threshold_chroma
                    or not unprocessed[y0, x0]
                ):
                    continue
                v = img[y0, x0]
                coef = vcomb_pinv @ v  # shape (2,)
                if coef[1] < 0:
                    coef[0] = _dot1d(vb_pinv[0], v)

                img_coeff[y0, x0] = coef
                img_df[y0, x0] = coef[0] * color_body
                unprocessed[y0, x0] = False


@njit(
    [
        types.void(
            types.Array(types.float32, 3, 'C'),
            types.Array(types.float32, 3, 'C'),
            types.Array(types.float32, 3, 'C'),
            types.Array(types.float32, 1, 'C'),  # illum
            types.Array(types.float32, 3, 'C'),
            types.float32,  # threshold_chroma
            types.Array(types.boolean, 2, 'C'),
            types.Array(types.boolean, 2, 'C'),
        )
    ],
    nogil=True,
    cache=True,
)
def _handle_combine(
    img: np.ndarray,
    img_df: np.ndarray,
    img_coeff: np.ndarray,
    illum: np.ndarray,
    chroma: np.ndarray,
    threshold_chroma: float,
    unprocessed: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    mask_combine: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
):
    h, w, _ = img.shape
    processed = np.bitwise_not(unprocessed)
    for y in range(h):
        for x in range(w):
            if unprocessed[y, x] != 0 or not mask_combine[y, x]:
                continue
            # Calculate chromaticity difference
            chroma_body = chroma[y, x]

            # Find diffuse pixel with closest chromaticity
            c_diff_sum = _ch_abs_sum(chroma - chroma_body)

            ind_y, ind_x = _masked_argmax(-c_diff_sum, processed, processed)

            color_body = img_df[ind_y, ind_x]
            chroma_body = chroma[ind_y, ind_x]
            vcomb = np.vstack((color_body, illum)).T

            # Get unprocessed pixel with similar chromaticity
            vcomb_pinv = np.linalg.pinv(vcomb)
            thresh = c_diff_sum[ind_y, ind_x] + 0.1 * threshold_chroma
            for y0 in range(h):
                for x0 in range(w):
                    if not unprocessed[y0, x0] or c_diff_sum[y0, x0] >= thresh:
                        continue
                    v = img[y0, x0]
                    coef = vcomb_pinv @ v

                    img_coeff[y0, x0] = coef
                    img_df[y0, x0] = coef[0] * color_body
                    unprocessed[y0, x0] = False
                    processed[y0, x0] = True


def shen2008(
    img: np.ndarray,
    threshold_chroma: float = 0.03,
):
    img = np.float32(img)
    height, width, _ = img.shape

    # Calculate specular-free image
    ch_mini = img.min(axis=-1)  # type: np.ndarray
    ch_maxi = img.max(axis=-1)  # type: np.ndarray
    ch_mini_mean = np.mean(ch_mini)

    img_msf = img - np.expand_dims(ch_mini, -1) + ch_mini_mean

    # Calculate the mask of combined pixels and diffuse pixels
    thresh = 2 * ch_mini_mean
    mask_combine = ch_mini > thresh
    mask_diffuse = (ch_maxi < thresh) & (ch_maxi > 20)

    # chromaticity
    chroma = img_msf / np.sum(img_msf, axis=-1, keepdims=True)
    chroma = np.float32(chroma)

    # Specularity removal
    # Find the pixels that need processing
    unprocessed = mask_combine | mask_diffuse

    # Illuminant, assumed white
    illum = np.array((255, 255, 255), dtype=np.float32)

    img_df = img.copy()

    # The diffuse and specular coefficient of each pixel
    img_coeff = np.zeros((height, width, 2), dtype=np.float32)
    _handle_diffuse(
        img,
        img_df,
        img_coeff,
        ch_maxi,
        illum,
        chroma,
        threshold_chroma,
        unprocessed,
        mask_diffuse,
    )
    _handle_combine(
        img,
        img_df,
        img_coeff,
        illum,
        chroma,
        threshold_chroma,
        unprocessed,
        mask_combine,
    )
    return img_df
