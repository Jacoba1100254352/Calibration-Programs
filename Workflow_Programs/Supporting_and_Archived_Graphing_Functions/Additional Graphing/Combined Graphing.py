import matplotlib.pyplot as plt

from Workflow_Programs.Configuration_Variables import *


def find_percentile_index(force_data, percentile):
	"""
	Find the index of the first occurrence of a value equal to or greater than the specified percentile.

	:param force_data: Array-like, data in which to find the percentile index.
	:param percentile: Integer, the desired percentile.
	:return: Tuple, (index, percentile_value).
	"""
	percentile_value = np.percentile(force_data, percentile)
	index = np.argmax(force_data >= percentile_value)
	return index


def read_arduino_data(filename, sensor_num):
	# Adjusted to read from CSV or TXT file based on file extension
	if str(filename).endswith('.txt'):
		with open(filename, "r") as file:
			lines = file.readlines()
		arduino_time = [(i + 1) * 20 for i in range(len(lines))]  # Assuming 20s intervals
		arduino_force = [float(line.split(",")[sensor_num].strip()) for line in lines]
	else:
		arduino_data = pd.read_csv(filename)
		arduino_time = arduino_data["Time [s]"]
		arduino_force = arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
	return arduino_time, arduino_force


def read_instron_data(filename, sensor_num):
	if str(filename).endswith('.xlsx'):
		excel_data = pd.read_excel(filename, sheet_name=f"Sensor {sensor_num}")
		excel_time = excel_data["Time [s]"].values * 1000  # Scaling time by 1000
		excel_force = [abs(value) for value in excel_data["Force [N]"].values]
	else:
		excel_data = pd.read_csv(filename)
		excel_time = excel_data["Time [s]"].values
		excel_force = excel_data["Force [N]"].values
	return excel_time, excel_force


def plot_sensor_data(time_data_1, force_data_1, time_data_2, force_data_2, sensor_num, label_1, label_2):
	fig, ax1 = plt.subplots(figsize=(10, 6))
	ax1.plot(time_data_1, force_data_1, label=label_1, color="orange")
	ax1.set_xlabel("Time [s]")
	ax1.set_ylabel(label_1, color="orange")
	ax1.tick_params(axis="y", labelcolor="orange")
	
	ax2 = ax1.twinx()
	ax2.plot(time_data_2, force_data_2, label=label_2, color="blue")
	ax2.set_ylabel(label_2, color="blue")
	ax2.tick_params(axis="y", labelcolor="blue")
	
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2, loc=0)
	
	plt.title(f"Overlay of Force Data for Sensor Set {SENSOR_SET}, Sensor {sensor_num}, Test {TEST_NUM}")
	plt.grid(True)
	plt.show()


# Running the analysis and plotting the results
for sensorNum in range(1, NUM_SENSORS + 1):
	# Arduino file path
	arduino_filename = get_data_filepath(ORIGINAL_ARDUINO_DIR, sensorNum)
	arduino_time, arduino_force = read_arduino_data(arduino_filename, sensorNum)
	
	# Instron file path (no sensor number required)
	instron_filename = get_data_filepath(ORIGINAL_INSTRON_DIR)
	instron_time, instron_force = read_instron_data(instron_filename, sensorNum)
	
	plot_sensor_data(arduino_time, arduino_force, instron_time, instron_force, sensorNum, "Arduino Force",
	                 "Instron Force")
