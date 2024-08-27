import matplotlib.pyplot as plt
import pandas as pd

from Configuration_Variables import *


def read_sensor_data(filename):
	"""
	Read sensor data from a CSV file.

	:param filename: Path, the file from which to read sensor data.
	:return: Tuple, (time_data, force_data).
	"""
	data = pd.read_csv(filename)
	time_data = data["Time [s]"].values
	try:
		force_data = data["Force [N]"].values
	except:
		force_data = data["Force2 [N]"].values
	
	return time_data, force_data


def plot_sensor_data(arduino_time, arduino_force, excel_time, excel_force, sensor_num):
	"""
	Plot and display the sensor data.

	:param arduino_time: Array-like, time data from Arduino.
	:param arduino_force: Array-like, force data from Arduino.
	:param excel_time: Array-like, time data from Instron.
	:param excel_force: Array-like, force data from Instron.
	:param sensor_num: Integer, the sensor number.
	"""
	fig, ax1 = plt.subplots(figsize=(10, 6))
	ax1.plot(arduino_time, arduino_force, label=f"Arduino Sensor {sensor_num}", color="orange")
	ax1.set_xlabel("Time [s]")
	ax1.set_ylabel("Arduino Force [N]", color="orange")
	ax1.tick_params(axis="y", labelcolor="orange")
	
	ax2 = ax1.twinx()
	ax2.plot(excel_time, excel_force, label=f"Instron Sensor {sensor_num}", color="blue")
	ax2.set_ylabel("Instron Force [N]", color="blue")
	ax2.tick_params(axis="y", labelcolor="blue")
	
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2, loc=0)
	
	plt.title(f"Overlay of Force Data for Sensor Set {SENSOR_SET}, Sensor {sensor_num}, Test {TEST_NUM}")
	plt.grid(True)
	plt.show()


# Running the analysis and plotting the results
for sensorNum in SENSORS_RANGE:
	arduino_filename = get_data_filepath(ALIGNED_ARDUINO_DIR, sensorNum)
	instron_filename = get_data_filepath(ALIGNED_INSTRON_DIR, sensorNum)
	
	arduino_time, arduino_force = read_sensor_data(arduino_filename)
	excel_time, excel_force = read_sensor_data(instron_filename)
	
	plot_sensor_data(arduino_time, -arduino_force, excel_time, excel_force, sensorNum)
