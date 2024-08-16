from pandas import DataFrame, read_csv, read_excel

from Configuration_Variables import *


NUM_SENSORS_OFFSET = 4
displacement = 0  # Displacement offset for aligning data if RTC is incorrect


def parse_arduino_data(line, sensor_num=None):
	"""
	Parse a line of Arduino data.

	Allows for the option to simplify the data to only include the ADC readings.

	:param line: String, a line from Arduino data file.
	:param sensor_num: Integer, the number of the sensor to be processed.
	:return: List, parsed values including timestamp, sensor readings, and total values.
	"""
	values = line.replace(",\t\t", ",").strip().split(",")
	
	if len(values) <= 11:
		print(f"Ignoring invalid values -> {values}")
		return []
	
	timestamp = [round(float(values[0]) / 1000, 2)]
	
	if SIMPLIFY:
		if sensor_num is not None and 0 <= sensor_num < len(values):
			adc_readings = [int(values[sensor_num])]
			force_values = []  # [float(values[sensor_num + NUM_SENSORS_OFFSET])]
		else:
			print("Invalid sensor number")
			return []
	else:
		adc_readings = list(map(int, values[1:5]))
		force_values = list(map(float, values[5:9]))
	
	total_values = [] if SIMPLIFY else list(map(float, values[9:11]))
	
	unix_time = [str(float(values[11]) + displacement)]
	
	return unix_time + timestamp + adc_readings + force_values + total_values


def process_arduino_data(sensor_num):
	"""
	Process Arduino data for a given sensor number and convert it to a DataFrame.

	:param sensor_num: Integer, the number of the sensor to be processed.
	:return: DataFrame, processed Arduino data.
	"""
	arduino_filename = get_data_filepath(ORIGINAL_ARDUINO_DIR, sensor_num)
	with open(arduino_filename, "r") as file:
		data = [parse_arduino_data(line, sensor_num) for line in file if parse_arduino_data(line, sensor_num)]
	
	if not data or not data[0]:
		print(f"No valid data found in {arduino_filename}")
		return DataFrame()
	
	num_columns = len(data[0])
	column_names = ["Timestamp"] + ["Time [s]"]
	if SIMPLIFY:
		column_names += [f"ADC"]
	else:
		column_names += [f"ADC{i}" for i in range(1, NUM_SENSORS_OFFSET + 1)] + [f"Force{i} [N]" for i in range(1, NUM_SENSORS_OFFSET + 1)]
		if num_columns > 9:
			column_names += ["TotalForce1 [N]", "TotalForce2 [N]"]
	
	return DataFrame(data, columns=column_names)


def process_instron_data(sensor_num):
	"""
	Process Instron data for a given sensor number.

	Simplify the data to only include the Force readings.

	:param sensor_num: Integer, the number of the sensor to be processed.
	:return: DataFrame, processed Instron data.
	"""
	instron_data_filename = get_data_filepath(ORIGINAL_INSTRON_DIR)
	# Lambda function for ignoring the Unnamed column and the Displacement column from the xlsx file
	column_omission = lambda x: x != 'Unnamed: 0' and x != 'Displacement [mm]'
	# NOTE: The "Timestamp" column will be included and only include 1 value at slot [0]
	instron_data = read_excel(instron_data_filename, sheet_name=f"Sensor {sensor_num}", usecols=column_omission)
	return instron_data


def process_and_save_csv(read_filename, write_filename, process_function, **kwargs):
	"""
	Read data from a CSV file, process it, and save it back to another CSV file.
	If read_filename is None, it directly processes the data using the process_function.

	:param read_filename: Path or None, the file from which to read data.
	:param write_filename: Path, the file to which processed data will be written.
	:param process_function: Function, the function to process the data.
	:param kwargs: Additional arguments passed to the process function.
	"""
	if read_filename is not None:
		data = read_csv(read_filename)
		processed_data = process_function(data, **kwargs)
	else:
		processed_data = process_function(**kwargs)
	
	processed_data.to_csv(write_filename, index=False)
	print(f"Data saved to {write_filename}")


def write_raw_data_to_csv(_displacement=0):
	"""
	Process raw data for each sensor and write it to CSV files.
	"""
	global displacement
	displacement = _displacement
	
	for sensor_num in SENSORS_RANGE:
		instron_csv_filename = get_data_filepath(PARSED_INSTRON_DIR, sensor_num)
		process_and_save_csv(None, instron_csv_filename, process_instron_data, sensor_num=sensor_num)
		
		arduino_csv_filename = get_data_filepath(PARSED_ARDUINO_DIR, sensor_num)
		process_and_save_csv(None, arduino_csv_filename, process_arduino_data, sensor_num=sensor_num)
