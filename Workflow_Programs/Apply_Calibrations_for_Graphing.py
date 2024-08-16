from pandas import DataFrame, read_csv

from Configuration_Variables import *


def read_coefficients(filename):
	"""
	Read coefficients from a CSV file.

	:param filename: Path, the file from which to read coefficients.
	:return: List of tuples, containing the coefficients.
	"""
	try:
		# Read the coefficients from the CSV file
		df = read_csv(filename)
		
		# Convert the DataFrame to a list of tuples
		coefficients = list(df[['m', 'b']].itertuples(index=False, name=None))
		
		return coefficients
	except FileNotFoundError:
		print(f"Warning: The file {filename} does not exist!")
		raise
	except ValueError as e:
		print(e)
		raise


def write_updated_data_to_csv(filename, data):
	"""
	Write sensor data to a CSV file.

	:param filename: Path, the file to which the data will be written.
	:param data: DataFrame, the sensor data to be written.
	"""
	columns = ["Time [s]", "Force [N]"] if SIMPLIFY else ["Time [s]", "ADC1", "ADC2", "ADC3", "ADC4",
	                                                      "Force1 [N]", "Force2 [N]", "Force3 [N]", "Force4 [N]",
	                                                      "TotalForce1 [N]", "TotalForce2 [N]"]
	df = DataFrame(data, columns=columns)
	df.to_csv(filename, index=False)
	print(f"Data successfully saved to {filename}")


def apply_calibration_coefficients():
	"""
	Apply calibration coefficients to each sensor's data and center the data around zero.
	"""
	# Read the new calibration coefficients
	new_coefficients = read_coefficients(get_data_filepath(COEFFICIENTS_DIR))
	
	# Convert the coefficients to a dictionary
	coefficients_dict = {sensor_num: coeff for sensor_num, coeff in zip(SENSORS_RANGE, new_coefficients)}
	
	# Apply the calibration to each sensor's data
	for sensor_num in SENSORS_RANGE:
		# Read the aligned data
		aligned_data_filename = get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num)
		aligned_arduino_data = read_csv(aligned_data_filename)
		calibrated_arduino_data = aligned_arduino_data.copy()
		
		# Apply calibration and center the data
		if SIMPLIFY:
			# Apply the calibration to the force data
			m, b = coefficients_dict[sensor_num]
			calibrated_arduino_data["Force [N]"] = m * calibrated_arduino_data["ADC"] + b
		else:
			# Apply the calibration to each force sensor and calculate the total force
			# This is different from SENSORS_RANGE because it's regardless of other sensors tested
			for _sensor_num in SENSORS_RANGE:
				m, b = coefficients_dict[_sensor_num]
				calibrated_arduino_data[f"Force{_sensor_num} [N]"] = m * calibrated_arduino_data[
					f"ADC{_sensor_num}"] + b
			calibrated_arduino_data["TotalForce1 [N]"] = sum(
				[calibrated_arduino_data[f"Force{_sensor_num} [N]"] for _sensor_num in SENSORS_RANGE])
			calibrated_arduino_data["TotalForce2 [N]"] = 0
		
		# Write the updated data to a CSV file
		updated_csv_filename = get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num)
		write_updated_data_to_csv(updated_csv_filename, calibrated_arduino_data)
