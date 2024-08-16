import numpy as np
from pandas import DataFrame, read_csv

from Configuration_Variables import *


def calculate_linear_fit(instron_force, arduino_raw_force):
	"""
	Calculate the linear regression coefficients for force data.

	This function computes the linear regression coefficients (slope 'm' and intercept 'c')
	that best fit the relationship between force data from an Excel file (typically Instron data)
	and raw force data from an Arduino device.

	:param instron_force: Array-like, force data from Excel (Instron data).
	:param arduino_raw_force: Array-like, raw force data from Arduino.
	:return: Tuple, (m, b) coefficients from the linear fit.
	"""
	A = np.vstack([arduino_raw_force, np.ones(len(arduino_raw_force))]).T
	m, b = np.linalg.lstsq(A, instron_force, rcond=None)[0]
	return m, b


def write_coefficients_to_file(filename, coefficients):
	"""
	Write the calculated linear regression coefficients to a CSV.

	This function takes the calculated coefficients and writes them into the specified file.
	The coefficients are formatted as a series of tuples, each representing the slope 'm'
	and intercept 'b' for a sensor.

	The formatting is meant to then be paste-able into the Arduino code.

	:param filename: String, path of the file where coefficients are to be saved.
	:param coefficients: Iterable of tuples, each tuple containing the 'm' and 'c' coefficients.
	"""
	
	# Create a DataFrame with SENSOR_RANGE as the index
	df = DataFrame(coefficients, index=SENSORS_RANGE, columns=['m', 'b'])
	
	# Generate the formatted data for each sensor using the respective coefficients in each row
	df['formatted_data'] = df.apply(lambda row: f"{{ {row['m']}, {row['b']} }}", axis=1)
	
	# Create a dictionary to map sensor number to formatted data
	sensor_data_dict = {sensor: f"{{ {m}, {b} }}" for (sensor, (m, b)) in zip(SENSORS_RANGE, coefficients)}
	
	# Generate the fully formatted construction
	formatted_construction = []
	for sensor_num in range(1, 5):
		if sensor_num in sensor_data_dict:
			formatted_construction.append(sensor_data_dict[sensor_num])
		else:
			formatted_construction.append('')
	
	formatted_construction_str = f"{{ {', '.join(formatted_construction)} }}"
	df['fully_formatted_construction'] = formatted_construction_str
	
	# Save to CSV
	df.to_csv(filename, index=False)
	
	# Print formatted data and fully formatted construction for verification
	formatted_data = f"{', '.join([f'{{ {m}, {b} }}' for m, b in coefficients])}"
	
	# If there are all 4 sensors, wrap the data in curly braces
	if len(coefficients) == 4:
		formatted_data = f"{{ {formatted_data} }}"
	
	print(f"New coefficients: {formatted_data}")


def calculate_coefficients():
	"""
	Calculate new calibration coefficients for all sensors and write them to a file.
	"""
	new_coefficients_corrected = []
	
	for sensor_num in SENSORS_RANGE:
		# Read data from CSV files
		instron_data = read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
		aligned_arduino_data = read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num))
		
		# Get force data from both datasets
		instron_force = instron_data["Force [N]"].values
		arduino_raw_force = aligned_arduino_data[f"ADC{'' if SIMPLIFY else sensor_num}"].values
		
		# Calculate linear fit and append to list of new coefficients
		new_coefficients_corrected.append(calculate_linear_fit(instron_force, arduino_raw_force))
	
	write_coefficients_to_file(get_data_filepath(COEFFICIENTS_DIR), new_coefficients_corrected)
