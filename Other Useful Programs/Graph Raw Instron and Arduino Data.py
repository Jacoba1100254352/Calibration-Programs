import matplotlib.pyplot as plt
import pandas as pd

from Configuration_Variables import *


def read_arduino_data(filename, sensor_num):
	"""
	Read Arduino data from a text file.

	:param filename: Path, the file from which to read Arduino data.
	:param sensor_num: Integer, the number of the sensor.
	:return: Tuple, (arduino_time, arduino_force).
	"""
	with open(filename, "r") as file:
		lines = file.readlines()
	arduino_time = [(i + 1) * 20 for i in range(len(lines))]  # Assuming 20s intervals
	arduino_force = [float(line.split(",")[sensor_num].strip()) for line in lines]
	return arduino_time, arduino_force


def read_instron_data(filename, sheet_name):
	"""
	Read Instron data from an Excel file.

	:param filename: Path, the file from which to read Instron data.
	:param sheet_name: String, the name of the sheet to read.
	:return: Tuple, (excel_time, excel_force).
	"""
	excel_data = pd.read_excel(filename, sheet_name=sheet_name)
	excel_time = excel_data["Time [s]"].values * 1000  # Assuming time needs to be scaled
	excel_force = [abs(value) for value in excel_data["Force [N]"].values]
	return excel_time, excel_force


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
	ax1.set_xlabel("Time [ms]" if excel_time[-1] > arduino_time[-1] else "Time [s]")
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
	arduino_filename = get_data_filepath(ORIGINAL_ARDUINO_DIR, sensorNum)
	excel_filename = get_data_filepath(ORIGINAL_INSTRON_DIR)
	
	arduino_time, arduino_force = read_arduino_data(arduino_filename, sensorNum)
	excel_time, excel_force = read_instron_data(excel_filename, f"Sensor {sensorNum}")
	
	plot_sensor_data(arduino_time, arduino_force, excel_time, excel_force, sensorNum)
