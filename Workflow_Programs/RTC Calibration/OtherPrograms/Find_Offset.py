import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Workflow_Programs.Configuration_Variables import *


def find_percentile_index(force_data, percentile):
	percentile_value = np.percentile(force_data, percentile)
	index = np.argmax(force_data >= percentile_value)
	return index


def calculate_avg_time_offset(parsed_instron_data, parsed_arduino_data):
	percentiles = range(1, 100, 1)
	time_offsets = []
	
	for percentile in percentiles:
		instron_index = find_percentile_index(-parsed_instron_data["Force [N]"], percentile)
		arduino_index = find_percentile_index(parsed_arduino_data[f"ADC{'' if SIMPLIFY else sensor_num}"], percentile)
		
		time_offset = parsed_instron_data["Time [s]"][instron_index] - parsed_arduino_data["Time [s]"][arduino_index]
		time_offsets.append(time_offset)
	
	return sum(time_offsets) / len(time_offsets)


def align_data_with_offset(parsed_instron_data, parsed_arduino_data, time_offset, diff_offset):
	parsed_arduino_data["Time [s]"] += time_offset + diff_offset
	
	min_length = min(len(parsed_instron_data), len(parsed_arduino_data))
	return parsed_instron_data.head(min_length), parsed_arduino_data.head(min_length)


def calculate_linear_fit(excel_force, arduino_raw_force):
	A = np.vstack([arduino_raw_force, np.ones(len(arduino_raw_force))]).T
	m, b = np.linalg.lstsq(A, excel_force, rcond=None)[0]
	return m, b


def calculate_coefficients(instron_data, aligned_arduino_data):
	instron_force = instron_data["Force [N]"].values
	arduino_raw_force = aligned_arduino_data[f"ADC{'' if SIMPLIFY else sensor_num}"].values
	return calculate_linear_fit(instron_force, arduino_raw_force)


def apply_calibration_coefficients(aligned_arduino_data, new_coefficients):
	calibrated_data = aligned_arduino_data.copy()
	if SIMPLIFY:
		m, b = new_coefficients[:]
		calibrated_data["Force [N]"] = m * calibrated_data["ADC"] + b
	else:
		for _sensor_num in range(1, NUM_SENSORS + 1):
			m, b = new_coefficients[_sensor_num - 1][:]
			calibrated_data[f"Force{_sensor_num} [N]"] = m * calibrated_data[f"ADC{_sensor_num}"] + b
		calibrated_data["TotalForce1 [N]"] = sum(
			[calibrated_data[f"Force{_sensor_num} [N]"] for _sensor_num in range(1, NUM_SENSORS + 1)])
		calibrated_data["TotalForce2 [N]"] = 0
	return calibrated_data


def calculate_difference(instron_data, updated_arduino_data, sensor_num):
	# Ensure that 'Force [N]' or the appropriate 'Force{sensor_num} [N]' column exists in the DataFrame
	instron_force = instron_data["Force [N]"]
	
	# Determine the appropriate column names based on SIMPLIFY flag
	force_column = "Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"
	adc_column = "ADC" if SIMPLIFY else f"ADC{sensor_num}"
	
	# Assuming updated_arduino_data is a slice from another DataFrame
	updated_arduino_data = updated_arduino_data.copy()
	
	# Using .loc for assignment to address SettingWithCopyWarning
	updated_arduino_data.loc[:, force_column] = updated_arduino_data[adc_column]
	
	updated_arduino_force = updated_arduino_data[force_column]
	return instron_force - updated_arduino_force


def find_best_offset(parsed_instron_data, parsed_arduino_data, avg_time_offset, sensor_num):
	best_offset = None
	min_variance = float('inf')
	
	for diff_offset in range(-50, 51, 1):
		aligned_instron, aligned_arduino = align_data_with_offset(parsed_instron_data, parsed_arduino_data,
		                                                          avg_time_offset, diff_offset)
		difference = calculate_difference(aligned_instron, aligned_arduino, sensor_num)
		
		variance = np.var(difference)
		if variance < min_variance:
			min_variance = variance
			best_offset = diff_offset
	
	return best_offset, min_variance


# Main Execution
best_offsets = {}

for sensor_num in range(1, NUM_SENSORS + 1):
	# Load data (replace with actual data loading method)
	parsed_instron_data = pd.read_csv(get_data_filepath(PARSED_INSTRON_DIR, sensor_num))
	parsed_arduino_data = pd.read_csv(get_data_filepath(PARSED_ARDUINO_DIR, sensor_num))
	
	# Align data and calculate average time offset
	avg_time_offset = calculate_avg_time_offset(parsed_instron_data, parsed_arduino_data)
	
	# Find best offset
	best_offset, variance = find_best_offset(parsed_instron_data, parsed_arduino_data, avg_time_offset, sensor_num)
	best_offsets[sensor_num] = best_offset
	
	# Align data using best offset
	aligned_instron_data, aligned_arduino_data = align_data_with_offset(parsed_instron_data, parsed_arduino_data,
	                                                                    avg_time_offset, best_offset)
	
	# Calculate coefficients
	coefficients = calculate_coefficients(aligned_instron_data, aligned_arduino_data)
	
	# Apply coefficients
	calibrated_data = apply_calibration_coefficients(aligned_arduino_data, coefficients)
	
	# Calculate final difference
	final_difference = calculate_difference(aligned_instron_data, calibrated_data, sensor_num)
	
	# Extract time and force data from Instron and calibrated Arduino data
	instron_time = aligned_instron_data["Time [s]"]
	instron_force = aligned_instron_data["Force [N]"]
	arduino_time = calibrated_data["Time [s]"]
	arduino_force = calibrated_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
	
	# Setup plot
	plt.figure(figsize=(10, 6))
	plt.plot(arduino_time, arduino_force, label="Calibrated Arduino Data", color="red")
	plt.plot(instron_time, instron_force, label="Instron Data", color="blue")
	
	# Calculate and plot the difference after calibration
	difference = instron_force - arduino_force
	plt.plot(instron_time, difference, label="Difference After Calibration (Instron - Arduino)", color="green",
	         linestyle="--")
	
	# Set plot labels and title
	plt.xlabel("Time [s]")
	plt.ylabel("Force [N]")
	plt.legend()
	plt.title(f"Comparison of Force Data for Sensor {sensor_num} with Best Offset")
	plt.grid(True)
	
	# Save and show the plot
	plt.show()

# best_offsets now contains the optimal diff_offset for each sensor
print("Best offsets for each sensor:", best_offsets)
