import numpy as np
from hyperspy.drawing._markers import point

from skimage.feature import blob_dog, blob_log, blob_doh
from scipy.ndimage import gaussian_filter
from skimage.feature.peak import peak_local_max
from skimage.feature.blob import _prune_blobs

def _correlation(z, axis=0, mask=None, wrap=True, normalize_axes=None):
    r"""A generic function for applying a correlation with a mask.

    Takes a nd image and then preforms a auto-correlation on some axis.
    Used in the electron correlation and angular correlation codes. Uses
    the fft to speed up the correlation.

    Parameters
    ----------
    z: np.array
        A nd numpy array
    axis: int
        The axis to apply the correlation to
    mask: np.array
        A boolean array of the same size as z
    wrap: bool
        Allow the function to wrap or add zeros to the beginning and the
        end of z on the specified axis
    normalize: bool
        Subtract <I(\theta)>^2 and divide by <I(\theta)>^2
    """
    eps = np.finfo(float).eps # v small number
    if wrap is False:
        z_shape = np.shape(z)
        padder = [(0, 0)] * len(z_shape)
        pad = z_shape[axis]  # This will be faster if the length of the axis
        # is a power of 2.  Based on the numpy implementation.  Not terribly
        # faster I think..
        padder[axis] = (pad, pad)
        slicer = [slice(None),] * len(z_shape)
        slicer[axis] = slice(0, -2 * pad)  # creating the proper slices
        if mask is None:
            mask = np.zeros(shape=np.shape(z))
        z = np.pad(z, padder, "constant")

    if mask is not None or wrap is False:  # Need to scale for wrapping properly
        m = np.array(mask, dtype=bool)  # make sure it is a boolean array
        # this is to determine how many of the variables were non zero... This is really dumb.  but...
        # it works and I should stop trying to fix it (wreak it)
        mask_boolean = ~m  # inverting the boolean mask
        if wrap is False:  # padding with zeros to the function along some axis
            m = np.pad(
                m, padder, "constant"
            )  # all the zeros are masked (should account for padding
            #  when normalized.
            mask_boolean = np.pad(mask_boolean, padder, "constant")
        mask_fft = np.fft.rfft(mask_boolean, axis=axis)
        number_unmasked = np.fft.irfft(
            mask_fft * np.conjugate(mask_fft), axis=axis
        ).real
        number_unmasked[number_unmasked < 1] = 1  # get rid of divide by zero error for completely masked rows
        z[m] = 0

    # fast method uses a FFT and is a process which is O(n) = n log(n)
    I_fft = np.fft.rfft(z, axis=axis)
    a = np.fft.irfft(I_fft * np.conjugate(I_fft), axis=axis)

    if mask is not None:
        a = np.divide(a, number_unmasked)
        a[number_unmasked == 1] = 0
    else:
        a = np.divide(a, np.shape(z)[axis])

    if normalize_axes is not None:  # simplified way to calculate the normalization
        # Need two row mean's for the case when row mean = 0.  I don't know if that
        num_not_zero = np.sum(number_unmasked !=1, axis=normalize_axes)
        row_mean1 = np.divide(np.sum(a, axis=normalize_axes), num_not_zero)
        row_mean2 =row_mean1
        row_mean2[row_mean2 == 0] = 1
        row_mean1 = np.expand_dims(row_mean1, axis=normalize_axes)
        row_mean2 = np.expand_dims(row_mean2, axis=normalize_axes)
        a = np.divide(np.subtract(a, row_mean1), row_mean2)
        a[number_unmasked <5 ] = 0


    if wrap is False:
        a = a[slicer]
    return a


def _power(z, axis=0, mask=None, wrap=True, normalize=True):
    """The power spectrum of the correlation.

    This method is a little more complex if mask is not None due to
    the extra calculations necessary to ignore some of the pixels
    during the calculations.

    Parameters
    ----------
    z: np.array
        Some n-d array to get the power spectrum from.
    axis: int
        The axis to preform the operation on.
    mask: np.array
        A boolean mask to be applied.
    wrap: bool
        Choose if the function should wrap.  In most cases this will be True
        when calculating the power of some function
    normalize: bool
        Choose to normalize the function by the mean.

    Returns
    -------
    power: np.array
        The power spectrum along some axis
    """
    if mask is None:  # This might not normalize things as well
        I_fft = np.fft.rfft(z, axis=axis)
        return (I_fft * np.conjugate(I_fft)).real
    else:
        return np.power(
            np.fft.rfft(
                _correlation(z=z, axis=axis, mask=mask, wrap=wrap, normalize=normalize)
            ),
            2,
        ).real


def corr_to_power(z):
    return np.power(np.fft.rfft(z, axis=1), 2).real


def wrap_set_float(list, bottom, top, value):
    """This function sets values in a list assuming that
    the list is circular and allows for float bottom and float top
    which are equal to the residual times that value.
    """
    ceiling_bottom = int(np.ceil(bottom))
    residual_bottom = ceiling_bottom-bottom
    floor_top = int(np.floor(top))
    residual_top = top-floor_top
    if floor_top > len(list) - 1:
        list[ceiling_bottom:] = value
        new_floor_top = floor_top % len(list)
        list[new_floor_top] = value
        list[new_floor_top+1] = value*residual_top
    elif ceiling_bottom < 0:
        list[:floor_top] = value
        list[ceiling_bottom:] = value
        list[ceiling_bottom-1] = value*residual_bottom
    else:
        list[ceiling_bottom:floor_top+1] = value
        list[ceiling_bottom-1] = value*residual_bottom
        if floor_top + 1 > len(list) - 1:
            list[0] = value*residual_top
        else:
            list[floor_top+1] = value*residual_top
    return list


def get_interpolation_matrix(angles, angular_range, num_points, method="average"):
    if method is "average":
        angular_ranges = [(angle - angular_range / (2*np.pi), angle + angular_range / (2*np.pi)) for angle in angles]
        angular_ranges = np.multiply(angular_ranges, num_points)
        interpolation_matrix = np.zeros(num_points)
        for i, angle in enumerate(angular_ranges):
            wrap_set_float(interpolation_matrix, top=angle[1], bottom=angle[0], value=1)
        return interpolation_matrix
    else:
        angular_ranges = [(angle - angular_range / (2*np.pi), angle + angular_range / (2*np.pi)) for angle in angles]
        angular_ranges = np.multiply(angular_ranges, num_points)
        interpolation_matrix = np.zeros((len(angles),num_points))
        for i, angle in enumerate(angular_ranges):
            wrap_set_float(interpolation_matrix[i,:], top=angle[1], bottom=angle[0], value=1)
        return interpolation_matrix

def symmetry_stem(signal, interpolation, method="average"):
    if method is "average":
        return np.matmul(signal, np.transpose(interpolation))
    if method is "max":
        val = [np.amax([np.matmul(signal, i)for i in interp], axis=0)
               for interp in interpolation]
        return val



def blob_finding(data, method, **kwargs):
    """This method helps to format the output from the blob methods
    in skimage for a more hyperspy like format using hs.markers
    """
    method_dict = {"log": blob_log, "dog": blob_dog, "doh": blob_doh}
    points = method_dict[method](data, **kwargs)
    return points


def peak_finding(data, sigma, overlap=0.5, **kwargs):
    """This method helps to format the output from the blob methods
    in skimage for a more hyperspy like format using hs.markers
    """
    print("data shape", np.shape(data))
    local_maxima = peak_local_max(data,
                                  footprint=np.ones((3,) * (data.ndim)),
                                  **kwargs)
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 4))
        # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma

    sigmas_of_peaks = sigma[local_maxima[:, 0]]
    # Remove sigma index and replace with sigmas
    #lm[:, 0] = sigmas_of_peaks
    #pruned = _prune_blobs(lm, overlap, sigma_dim=3)
    return lm






