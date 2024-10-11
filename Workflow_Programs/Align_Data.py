from Configuration_Variables import *
from Workflow_Programs.Supplemental_Sensor_Graph_Functions import apply_smoothing


displacement = 0


def find_initial_timestamp_offset(instron_timestamp, arduino_data):
	"""
	Calculate the initial offset between the Instron and Arduino datasets
	using the first timestamp of the Instron data and the nearest Arduino timestamp.

	:param instron_timestamp: Float, the timestamp from the Instron data.
	:param arduino_data: DataFrame, the Arduino data including "Timestamp" for each data point.
	:return: Float, the initial time offset.
	"""
	# Find the Arduino timestamp closest to the Instron timestamp
	arduino_timestamps = arduino_data["Timestamp"] + displacement
	closest_index = (np.abs(arduino_timestamps - instron_timestamp)).argmin()
	closest_timestamp = arduino_timestamps.iloc[closest_index]
	
	# Calculate the initial offset
	initial_offset = instron_timestamp - closest_timestamp
	return initial_offset, closest_index


def align_and_save_data(sensor_num, parsed_instron_data, parsed_arduino_data):
	"""
	Align and save Arduino data to the specified directory, using the new "Timestamp" columns for initial alignment.

	:param sensor_num: Integer, the sensor number.
	:param parsed_instron_data: DataFrame, the data from the Instron device.
	:param parsed_arduino_data: DataFrame, the data from the Arduino device.
	"""
	# Get the initial timestamp from the Instron data
	instron_initial_timestamp = parsed_instron_data["Timestamp"].iloc[0]
	
	# Calculate the initial offset and the index of the closest Arduino timestamp
	initial_offset, closest_arduino_index = find_initial_timestamp_offset(instron_initial_timestamp, parsed_arduino_data)
	
	# Drop the Arduino data points before the closest index
	parsed_arduino_data = parsed_arduino_data.iloc[closest_arduino_index:].copy()
	
	# Reset Arduino "Time [s]" to start at 0 and round to the nearest 0.02
	parsed_arduino_data.loc[:, "Time [s]"] = np.round((parsed_arduino_data["Time [s]"] - parsed_arduino_data["Time [s]"].iloc[
		0]) / 0.02) * 0.02
	
	# Round the ADC values to the nearest whole numbers
	adc_columns = [col for col in parsed_arduino_data.columns if 'ADC' in col]
	parsed_arduino_data.loc[:, adc_columns] = parsed_arduino_data[adc_columns].round(0)
	
	# Drop the "Timestamp" column as it's no longer needed
	# parsed_arduino_data = parsed_arduino_data.drop(columns=["Timestamp"])
	
	# Truncate both datasets to the shortest length
	min_length = min(len(parsed_instron_data), len(parsed_arduino_data))
	aligned_instron_data = parsed_instron_data.head(min_length)
	aligned_arduino_data = parsed_arduino_data.head(min_length)
	
	# Smooth arduino ADC values
	# aligned_arduino_data.loc[:, adc_columns] = apply_smoothing(aligned_arduino_data[adc_columns], "boxcar", 100, None)
	aligned_arduino_data.loc[:, adc_columns] = apply_smoothing(
		aligned_arduino_data[adc_columns].copy(), "boxcar", 100, None).astype(np.int64)
	
	# Get the aligned data directory names
	aligned_arduino_data_dir = get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num)
	aligned_instron_data_dir = get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num)
	
	# Save the aligned data
	aligned_arduino_data.to_csv(aligned_arduino_data_dir, index=False)
	aligned_instron_data.to_csv(aligned_instron_data_dir, index=False)
	print(f"Aligned Arduino data saved to {aligned_arduino_data_dir}")
	print(f"Aligned Instron data saved to {aligned_instron_data_dir}")


def align_data(_displacement=0):
	"""
	Align data for all sensors using the new alignment method.
	"""
	global displacement
	displacement = _displacement
	
	for sensor_num in SENSORS_RANGE:
		parsed_instron_data = pd.read_csv(get_data_filepath(PARSED_INSTRON_DIR, sensor_num))
		parsed_arduino_data = pd.read_csv(get_data_filepath(PARSED_ARDUINO_DIR, sensor_num))
		
		align_and_save_data(sensor_num, parsed_instron_data, parsed_arduino_data)
