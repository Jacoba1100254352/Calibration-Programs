import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
from scipy.stats import linregress


def avg(lst):
	return sum(lst) / len(lst)


def difference_polarity(lst1, lst2):
	return (avg(lst1) - avg(lst2)) / abs(avg(lst1) - avg(lst2))


def calculate_line_of_best_fit(x, y, isPolyfit=False, order=1):
	if not isPolyfit:
		slope_avg, intercept_avg, _, _, _ = linregress(x, y)
		line_of_best_fit = slope_avg * x + intercept_avg
	else:
		coefficients = np.polyfit(x, y, order)
		polynomial = np.poly1d(coefficients)
		line_of_best_fit = polynomial(x)
	return line_of_best_fit


# Function to apply smoothing to residuals
def apply_smoothing(residuals, method, window_size, poly_order):
	"""
	Apply smoothing to the residuals using the specified method.

	Parameters:
	- residuals: DataFrame or Series, the residual data to be smoothed.
	- method: The smoothing method ('savgol', 'boxcar', 'median', or None).
	- window_size: The window size for the smoothing operation.
	- poly_order: The polynomial order for Savitzky-Golay filter (only used if method is 'savgol').

	Returns:
	- smoothed_residuals: The smoothed residuals.
	"""
	
	def smooth_column(data_column, window_size):
		""" Helper function to smooth individual columns. """
		if method == 'savgol':
			if window_size is None:
				raise ValueError("Window size must be specified for Savitzky-Golay smoothing.")
			return savgol_filter(data_column, window_length=window_size, polyorder=poly_order)
		elif method == 'boxcar':
			if window_size is None:
				raise ValueError("Window size must be specified for boxcar smoothing.")
			smoothed = np.convolve(data_column, np.ones(window_size) / window_size, mode='valid')
			smoothed = np.pad(smoothed, (window_size // 2, window_size // 2), mode='edge')
			if len(smoothed) > len(data_column):
				return smoothed[:len(data_column)]
			elif len(smoothed) < len(data_column):
				return np.pad(smoothed, (0, len(data_column) - len(smoothed)), 'edge')
		elif method == 'median':
			if window_size is None:
				raise ValueError("Window size must be specified for median filtering.")
			if window_size % 2 == 0:  # Ensure window_size is odd
				window_size += 1
			return medfilt(data_column, kernel_size=window_size)
		else:
			return data_column
	
	# If the input is a DataFrame, apply smoothing to each column
	if isinstance(residuals, pd.DataFrame):
		for col in residuals.columns:
			residuals[col] = smooth_column(residuals[col].values.flatten(), window_size)
	else:
		residuals = smooth_column(residuals, window_size)
	
	return residuals
