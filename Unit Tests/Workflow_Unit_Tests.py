import unittest

import numpy as np

from Align_Data import find_percentile_index
from Calculate_Linear_Fit_Sensor_Calibrations import calculate_linear_fit


class TestFindPercentileIndex(unittest.TestCase):
	def test_percentile_index(self):
		data = [1, 3, 5, 7, 9]
		percentile = 50
		expected_index = 2
		expected_value = 5
		index, value = find_percentile_index(data, percentile)
		self.assertEqual(index, expected_index)
		self.assertEqual(value, expected_value)


class TestCalculateLinearFit(unittest.TestCase):
	def test_linear_fit(self):
		excel_force = np.array([2, 4, 6, 8])
		arduino_raw_force = np.array([1, 2, 3, 4])
		expected_m, expected_c = 2, 0  # y = 2x
		m, c = calculate_linear_fit(excel_force, arduino_raw_force)
		self.assertAlmostEqual(m, expected_m)
		self.assertAlmostEqual(c, expected_c)


if __name__ == '__main__':
	unittest.main()
