import numpy as np
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler

def to_gaf_image(time_series: np.ndarray, image_size: int):
    """
    Converts a 1D time-series into a 2D Gramian Angular Field (GAF) image.

    Args:
        time_series (np.ndarray): A 1D numpy array representing the time-series.
        image_size (int): The desired output image size (height and width).

    Returns:
        np.ndarray: A 2D GAF image of shape (image_size, image_size).
    """
    # Ensure the time series is a 1D array
    if time_series.ndim > 1:
        time_series = time_series.flatten()

    # Scale the time series to the range [-1, 1] as required by GAF
    scaler = MinMaxScaler(feature_range=(-1, 1))
    ts_scaled = scaler.fit_transform(time_series.reshape(1, -1))

    # Resample the time series to match the desired image size
    # This is a simple linear interpolation. More sophisticated methods could be used.
    ts_resampled = np.interp(
        np.linspace(0, len(ts_scaled[0]) - 1, image_size),
        np.arange(len(ts_scaled[0])),
        ts_scaled[0]
    ).reshape(1, -1)

    # Create a GAF transformer and apply it
    gaf = GramianAngularField(image_size=image_size, method='summation')
    gaf_image = gaf.fit_transform(ts_resampled)

    # The output is (1, image_size, image_size), so we squeeze it to (image_size, image_size)
    return gaf_image.squeeze(0)
