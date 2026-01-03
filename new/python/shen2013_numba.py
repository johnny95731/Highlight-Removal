import numpy as np
from numba import njit, types


@njit(
    [
        types.float32(
            types.Array(types.float32, 1, 'C'),
            types.Array(types.float32, 1, 'C'),
        )
    ],
    nogil=True,
    cache=True,
)
def _dist(a, b):
    s = 0.0
    for i, val in enumerate(a):
        s += abs(val - b[i])
    return s


@njit(
    [
        types.Tuple((types.Array(types.uint16, 1, 'C'), types.int64))(
            types.Array(types.float32, 2, 'C'),
            types.Array(types.boolean, 1, 'C'),
            types.int64,
            types.float32,
            types.int64,
        )
    ],
    nogil=True,
    cache=True,
)
def pixel_clustering(
    img_ch_pseudo: np.ndarray,
    mask: np.ndarray,
    num_pixel: int,
    thresh_chroma: float,
    max_num_clust: int = 100,
):
    clust_mean = np.zeros((max_num_clust, 2), dtype=np.float32)
    counts = np.zeros(max_num_clust, dtype=np.uint32)
    done = np.zeros(num_pixel, dtype=np.bool_)
    clust = np.zeros(num_pixel, dtype=np.uint16)

    label = 0
    for i in range(num_pixel):
        if not done[i] and mask[i]:
            pivot = img_ch_pseudo[i]
            label += 1
            for j in range(i, num_pixel):
                if not done[j] and mask[j]:
                    dist = _dist(pivot, img_ch_pseudo[j])
                    if dist < thresh_chroma:
                        done[j] = True
                        clust[j] = label

    num_clust = label

    if num_clust > max_num_clust:
        return clust, num_clust

    for i in range(num_pixel):
        k = clust[i]
        if 0 < k <= num_clust:
            counts[k] += 1
            clust_mean[k, 0] += img_ch_pseudo[i, 0]
            clust_mean[k, 1] += img_ch_pseudo[i, 1]

    for k in range(1, num_clust + 1):
        if counts[k] > 0:
            clust_mean[k, 0] /= counts[k]
            clust_mean[k, 1] /= counts[k]

    for i in range(num_pixel):
        if mask[i]:
            pivot = img_ch_pseudo[i]
            dist_min = -np.inf
            label = 1
            for k in range(2, num_clust + 1):
                dist = _dist(pivot, clust_mean[k])
                if dist < dist_min:
                    dist_min = dist
                    label = k
            clust[i] = label

    return clust, num_clust


@njit(
    [
        types.Array(types.uint8, 3, 'C')(
            types.Array(types.float32, 3, 'C'),
            types.float32,
            types.int64,
            types.float32,
            types.float64,
        )
    ],
    nogil=True,
    cache=True,
)
def _shen2013(
    img: np.ndarray,
    thresh_chroma: float,
    max_num_clust: int,
    thresh_percent: float,
    eps: float,
):
    h, w, ch = img.shape
    num_pixel = h * w
    flatted = np.reshape(img, (num_pixel, ch))

    ch_mini = np.empty(flatted.shape[0], dtype=np.float32)
    ch_maxi = np.empty_like(ch_mini)
    img_ran = np.empty_like(ch_mini)
    mean_ch_mini = np.float64(0.0)
    for y, pixel in enumerate(flatted):
        _mini = float('inf')
        _maxi = float('-inf')
        for val in pixel:
            if val < _mini:
                _mini = val
            if val > _maxi:
                _maxi = val
        ch_mini[y] = _mini
        ch_maxi[y] = _maxi
        img_ran[y] = ch_maxi[y] - ch_mini[y]
        mean_ch_mini += ch_mini[y]
    mean_ch_mini = np.float32(mean_ch_mini / ch_mini.shape[0])

    mask = np.empty_like(ch_mini, dtype=np.bool)
    img_ch_pseudo = np.empty((ch_mini.shape[0], 2), dtype=np.float32)
    for y, pixel in enumerate(flatted):
        mask[y] = ch_mini[y] > mean_ch_mini
        if mask[y]:
            bias = mean_ch_mini - ch_mini[y]
            _sum = np.float32(0)
            _mini = float('inf')
            _maxi = float('-inf')
            for val in pixel:
                frgb = val + bias
                _sum += frgb
                if frgb < _mini:
                    _mini = frgb
                if frgb > _maxi:
                    _maxi = frgb
            img_ch_pseudo[y, 0] = _mini / _sum
            img_ch_pseudo[y, 1] = _maxi / _sum
        else:
            img_ch_pseudo[y] = 0

    clust, num_clust = pixel_clustering(
        img_ch_pseudo, mask, num_pixel, thresh_chroma, max_num_clust
    )

    ratio = np.zeros(num_pixel, dtype=np.float32)
    img_ratio = np.zeros(num_pixel, dtype=np.float32)
    for k in range(1, num_clust + 1):
        num = 0
        for i in range(num_pixel):
            if clust[i] == k and img_ran[i] > mean_ch_mini:
                ratio[num] = ch_maxi[i] / (img_ran[i] + eps)
                num += 1
        if num == 0:
            continue

        tmp = np.sort(ratio[:num])
        ratio_est = tmp[int(num * thresh_percent)]
        for i in range(num_pixel):
            if clust[i] == k:
                img_ratio[i] = ratio_est

    img_df = np.empty((h, w, ch), dtype=np.uint8)
    for i, pixel in enumerate(flatted):
        y, x = divmod(i, w)
        if mask[i] == 1:
            uvalue = ch_maxi[i] - img_ratio[i] * img_ran[i]
            specular = max(uvalue, 0)
            for c, val in enumerate(pixel):
                fvalue = (pixel[c] - specular) * 255
                img_df[y, x, c] = min(255.0, max(0.0, fvalue))
        else:
            for c, val in enumerate(pixel):
                fvalue = pixel[c] * 255
                img_df[y, x, c] = min(255.0, max(0.0, fvalue))
    res = img_df
    return res


def shen2013(
    img: np.ndarray,
    th_chroma: float = 0.3,
    max_num_clust: int = 100,
    thresh_percent: float = 0.5,
    eps: float = 1e-10,
):
    """
    Shen2013 I_d = Shen2013(I)
    You can optionally edit the code to use kmeans instead of the clustering
    function proposed by the author.

    This method should have equivalent functionality as
    `sp_removal.cpp` distributed by the author.

    See also SIHR, Shen2008, Shen2009.
    """
    if img.ndim < 3:
        raise ValueError(
            f'`img` mush be an RGB image: img.ndim = {img.ndim} < 3'
        )
    elif img.shape[-1] != 3:
        raise ValueError(
            f'`img` mush be an RGB image: channels = {img.shape[-1]} != 3'
        )
    res = _shen2013(
        np.float32(img) / 255,
        th_chroma,
        max_num_clust,
        thresh_percent,
        eps,
    )
    return res
