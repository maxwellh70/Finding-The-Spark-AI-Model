""" This code applies preprocessing functions on the IEEE GRSS ESD satellite data."""
import numpy as np
from scipy.ndimage import gaussian_filter


def per_band_gaussian_filter(img: np.ndarray, sigma: float = 1):
    """
    For each band in the image, apply a gaussian filter with the given sigma.

    Parameters
    ----------
    img : np.ndarray
        The image to be filtered.
    sigma : float
        The sigma of the gaussian filter.

    Returns
    -------
    np.ndarray
        The filtered image.
    """
    for i in range(img.shape[0]):
        img[i] = gaussian_filter(img[i], sigma)
    return img


def quantile_clip(img_stack: np.ndarray,
                  clip_quantile: float,
                  group_by_time=True
                  ) -> np.ndarray:
    """
    This function clips the outliers of the image stack by the given quantile.

    Parameters
    ----------
    img_stack : np.ndarray
        The image stack to be clipped.
    clip_quantile : float
        The quantile to clip the outliers by.

    Returns
    -------
    np.ndarray
        The clipped image stack.
    """
    if group_by_time:
        axis = (-2, -1)
    else:
        axis = (0, -2, -1)
    data_lower_bound = np.quantile(
        img_stack,
        clip_quantile,
        axis=axis,
        keepdims=True
        )
    data_upper_bound = np.quantile(
        img_stack,
        1-clip_quantile,
        axis=axis,
        keepdims=True
        )
    img_stack = np.clip(img_stack, data_lower_bound, data_upper_bound)

    return img_stack


def minmax_scale(img: np.ndarray, group_by_time=True):
    """
    This function minmax scales the image stack.

    Parameters
    ----------
    img : np.ndarray
        The image stack to be minmax scaled.
    group_by_time : bool
        Whether to group by time or not.

    Returns
    -------
    np.ndarray
        The minmax scaled image stack.
    """
    if group_by_time:
        axis = (-2, -1)
    else:
        axis = (0, -2, -1)
    img = img.astype(np.float32)
    min_val = img.min(axis=axis, keepdims=True)
    max_val = img.max(axis=axis, keepdims=True)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img


def brighten(img, alpha=0.13, beta=0):
    """
    Function to brighten the image.

    Parameters
    ----------
    img : np.ndarray
        The image to be brightened.
    alpha : float
        The alpha parameter of the brightening.
    beta : float
        The beta parameter of the brightening.

    Returns
    -------
    np.ndarray
        The brightened image.
    """
    return np.clip(alpha * img + beta, 0.0, 1.0)


def gammacorr(band, gamma=2):
    """
    This function applies a gamma correction to the image.

    Parameters
    ----------
    band : np.ndarray
        The image to be gamma corrected.
    gamma : float
        The gamma parameter of the gamma correction.

    Returns
    -------
    np.ndarray
        The gamma corrected image.
    """
    return np.power(band, 1/gamma)


def maxprojection_viirs(
        viirs_stack: np.ndarray,
        clip_quantile: float = 0.01
        ) -> np.ndarray:
    """
    This function takes a directory of VIIRS tiles and returns a single
    image that is the max projection of the tiles.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.

    Returns
    -------
    np.ndarray
    """
    for i in range(viirs_stack.shape[0]):
        viirs_data_lower_bound = np.quantile(
            viirs_stack[i, :, :, :],
            clip_quantile
            )
        viirs_data_upper_bound = np.quantile(
            viirs_stack[i, :, :, :],
            1-clip_quantile
            )
        viirs_stack[i, :, :, :] = np.clip(
            viirs_stack[i, :, :, :],
            viirs_data_lower_bound,
            viirs_data_upper_bound
            )

    # Calculate the max projection of the viirs_data_stack along the third axis
    # and assign it to the blank_array
    viirs_stack = np.max(viirs_stack, axis=0)
    viirs_stack = minmax_scale(viirs_stack)

    return viirs_stack


def preprocess_sentinel1(
        sentinel1_stack: np.ndarray,
        clip_quantile: float = 0.01,
        sigma=1
        ) -> np.ndarray:
    """
    In this function we will preprocess sentinel1. The steps for preprocessing
    are the following:
        - Convert data to dB (log scale)
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gaussian filter
        - Minmax scale
    """

    # convert data to dB
    sentinel1_stack = np.log10(sentinel1_stack)

    # clip outliers
    sentinel1_stack = quantile_clip(
        sentinel1_stack,
        clip_quantile=clip_quantile
        )
    sentinel1_stack = per_band_gaussian_filter(sentinel1_stack, sigma=sigma)
    sentinel1_stack = minmax_scale(sentinel1_stack)

    return sentinel1_stack


def preprocess_sentinel2(sentinel2_stack: np.ndarray,
                         clip_quantile: float = 0.1,
                         gamma: float = 2.2
                         ) -> np.ndarray:
    """
    In this function we will preprocess sentinel-2. The steps for
    preprocessing are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    sentinel2_stack = quantile_clip(
        sentinel2_stack,
        clip_quantile=clip_quantile,
        group_by_time=False
        )
    sentinel2_stack = gammacorr(sentinel2_stack, gamma=gamma)
    sentinel2_stack = minmax_scale(sentinel2_stack, group_by_time=False)

    return sentinel2_stack


def preprocess_landsat(
        landsat_stack: np.ndarray,
        clip_quantile: float = 0.05,
        gamma: float = 2.2
        ) -> np.ndarray:
    """
    In this function we will preprocess landsat. The steps for preprocessing
    are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    landsat_stack = quantile_clip(
        landsat_stack,
        clip_quantile=clip_quantile,
        group_by_time=False
        )
    landsat_stack = gammacorr(landsat_stack, gamma=gamma)
    landsat_stack = minmax_scale(landsat_stack, group_by_time=False)

    return landsat_stack


def preprocess_viirs(viirs_stack, clip_quantile=0.05) -> np.ndarray:
    """
    In this function we will preprocess viirs. The steps for preprocessing are
    the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Minmax scale
    """
    viirs_stack = quantile_clip(
        viirs_stack,
        clip_quantile=clip_quantile,
        group_by_time=True
        )
    viirs_stack = minmax_scale(viirs_stack, group_by_time=True)
    return viirs_stack