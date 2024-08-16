from pandas import read_csv

from Configuration_Variables import *


def truncate_and_save_data(sensor_num, force_threshold):
	"""
	Truncate and save Arduino and Instron data to the specified directory, using the force threshold.

	:param sensor_num: Integer, the sensor number.
	:param force_threshold: Float, the force value to truncate the data at.
	"""
	# Load the aligned data
	aligned_instron_data = read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
	aligned_arduino_data = read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num))
	
	# Find the index where the Instron force first reaches the threshold
	truncate_index = aligned_instron_data['Force [N]'].ge(force_threshold).idxmin()
	
	# Truncate both datasets to this length
	truncated_instron_data = aligned_instron_data.iloc[:truncate_index]
	truncated_arduino_data = aligned_arduino_data.iloc[:truncate_index]
	
	# Save the truncated data, overwriting the aligned data
	truncated_instron_data.to_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num), index=False)
	truncated_arduino_data.to_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num), index=False)
	print(f"Truncated Arduino data saved to {get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num)}")
	print(f"Truncated Instron data saved to {get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num)}")


def truncate_data(force_threshold):
	"""
	Truncate data for all sensors using the specified force threshold.
	"""
	for sensor_num in SENSORS_RANGE:
		try:
			truncate_and_save_data(sensor_num, force_threshold)
		except Exception as e:
			print(f"Error truncating data for sensor {sensor_num}: {e}")
