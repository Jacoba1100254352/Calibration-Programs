from pathlib import Path

import numpy as np
import pandas as pd


# Test 3 sensor 2, Test 4 sensor 4, Test 5 sensor 2, Test 6 sensor 2, Test 7 sensor 2,

# (At the point of Test 8, the calibration coefficients were updated to the new values from Test 6 sensor 2)
# This will only affect the "Force [N]" and "TotalForce1 [N]" columns

###    Define Global constants   ###

# Sensor set and test number
SENSOR_SET = 1
TEST_NUM = 12
STARTING_SENSOR = 4

# Number of sensors to process
NUM_SENSORS = 1
SENSORS_RANGE = range(STARTING_SENSOR, STARTING_SENSOR + NUM_SENSORS)

# Working directory
WORKING_DIR = Path("../../Calibration Tests")

# Relative directory paths using pathlib
# Supplemental
SENSOR_SET_DIR = f"Sensor Set {SENSOR_SET}"
COEFFICIENTS_DIR = WORKING_DIR / "Coefficients" / SENSOR_SET_DIR
PLOTS_DIR = WORKING_DIR / "Data Plots" / SENSOR_SET_DIR

# Arduino
ARDUINO_DIR = "Arduino Data"
ALIGNED_ARDUINO_DIR = WORKING_DIR / ARDUINO_DIR / "Aligned Arduino Data" / SENSOR_SET_DIR
CALIBRATED_ARDUINO_DIR = WORKING_DIR / ARDUINO_DIR / "Calibrated Arduino Data" / SENSOR_SET_DIR
ORIGINAL_ARDUINO_DIR = WORKING_DIR / ARDUINO_DIR / "Original Arduino Data" / SENSOR_SET_DIR
PARSED_ARDUINO_DIR = WORKING_DIR / ARDUINO_DIR / "Parsed Arduino Data" / SENSOR_SET_DIR

# Instron
INSTRON_DIR = "Instron Data"
ALIGNED_INSTRON_DIR = WORKING_DIR / INSTRON_DIR / "Aligned Instron Data" / SENSOR_SET_DIR
ORIGINAL_INSTRON_DIR = WORKING_DIR / INSTRON_DIR / "Original Instron Data" / SENSOR_SET_DIR
PARSED_INSTRON_DIR = WORKING_DIR / INSTRON_DIR / "Parsed Instron Data" / SENSOR_SET_DIR

# Determines whether to only list the sensor specific values
SIMPLIFY = False


def get_data_filepath(directory, sensor_num=None, copy=False, _TEST_NUM=TEST_NUM):
	"""
	Generate the correct file path based on the given directory and sensor number.

	:param directory: The directory in which the file resides.
	:param sensor_num: The sensor number, if applicable.
	:param copy: Whether to append " 2" to the file name.
	:param _TEST_NUM: The test number to use in the file name.
	:return: The file path for the specified directory and sensor.
	"""
	
	# Combined mapping for extension and prefix
	mapping = {
		ORIGINAL_ARDUINO_DIR: ('txt', 'Original'),
		CALIBRATED_ARDUINO_DIR: ('csv', 'Updated'),
		ALIGNED_ARDUINO_DIR: ('csv', 'Aligned'),
		PARSED_ARDUINO_DIR: ('csv', 'Parsed'),
		ORIGINAL_INSTRON_DIR: ('xlsx', 'Original'),
		ALIGNED_INSTRON_DIR: ('csv', 'Aligned'),
		PARSED_INSTRON_DIR: ('csv', 'Parsed'),
		COEFFICIENTS_DIR: ('csv', ''),
		PLOTS_DIR: ('png', '')  # Special case handled in the return statement
	}
	
	if directory not in mapping:
		raise ValueError("Invalid directory")
	
	ext, prefix = mapping[directory]
	
	# Special case for COEFFICIENTS_DIR
	if directory == COEFFICIENTS_DIR:
		return directory / f"Test {_TEST_NUM} Calibration Coefficients.{ext}"
	
	# Special case for PLOTS_DIR
	if directory == PLOTS_DIR:
		if sensor_num is None:
			raise ValueError("Sensor number must be specified for PLOTS_DIR or COEFFICIENTS_DIR")
		else:
			return directory / f"Calibration Test {_TEST_NUM} Sensor {sensor_num} plot{'' if not copy else ' 2'}.{ext}"
	
	sensor_str = f" Sensor {sensor_num}" if sensor_num and directory != ORIGINAL_INSTRON_DIR else ""
	return directory / f"{prefix} Calibration Test {_TEST_NUM}{sensor_str} Data.{ext}"


def getTrainingData(_sensor_num, _TEST_RANGE):
	X_train = []
	y_train = []
	
	# Combine data from multiple tests into single X_train and y_train
	for _TEST_NUM in _TEST_RANGE:
		# Load data from CSV files
		instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num=_sensor_num, _TEST_NUM=_TEST_NUM))
		updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num=_sensor_num, _TEST_NUM=_TEST_NUM))
		
		# Ensure arrays are of equal length for accurate comparison
		min_length = min(len(instron_data), len(updated_arduino_data))
		instron_force = instron_data["Force [N]"].iloc[:min_length].values.reshape(-1, 1)
		updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force2 [N]"].iloc[:min_length].values.reshape(-1, 1)
		
		# Append to X_train and y_train
		X_train.append(instron_force)
		y_train.append(updated_arduino_force)
	
	# Convert lists to numpy arrays
	X_train = np.vstack(X_train)
	y_train = np.vstack(y_train)
	
	return X_train, y_train
