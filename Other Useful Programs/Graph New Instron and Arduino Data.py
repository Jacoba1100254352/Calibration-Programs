import matplotlib.pyplot as plt
import pandas as pd

from Configuration_Variables import *


def read_arduino_data(filename, sensor_num):
	"""
	Read Arduino data from a CSV file.

	:param filename: Path, the file from which to read Arduino data.
	:param sensor_num: Integer, the number of the sensor.
	:return: Tuple, (arduino_time, arduino_force).
	"""
	arduino_data = pd.read_csv(filename)
	arduino_time = arduino_data["Time [s]"]
	arduino_force = arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
	return arduino_time, arduino_force


def read_instron_data(filename, sensor_num):
	"""
	Read Instron data from an Excel file.

	:param filename: Path, the file from which to read Instron data.
	:param sensor_num: Integer, the number of the sensor.
	:return: Tuple, (excel_time, excel_force).
	"""
	excel_data = pd.read_excel(filename, sheet_name=f"Sensor {sensor_num}")
	excel_time = excel_data["Time [s]"].values * 1000
	excel_force = [abs(value) for value in excel_data["Force [N]"].values]
	return excel_time, excel_force


def plot_sensor_data(arduino_time, arduino_force, excel_time, excel_force, sensor_num):
	"""
	Plot and save the sensor data.

	:param arduino_time: Array-like, time data from Arduino.
	:param arduino_force: Array-like, force data from Arduino.
	:param excel_time: Array-like, time data from Instron.
	:param excel_force: Array-like, force data from Instron.
	:param sensor_num: Integer, the sensor number.
	"""
	plt.figure(figsize=(10, 6))
	plt.plot(arduino_time, arduino_force, label=f"Arduino Sensor {sensor_num}", color="orange")
	plt.plot(excel_time, excel_force, label=f"Instron Sensor {sensor_num}", color="blue")
	
	plt.xlabel("Time [s]")
	plt.ylabel("Force [N]")
	plt.legend()
	plt.title(f"Overlay of Force Data for Sensor Set {SENSOR_SET}, Sensor {sensor_num}, Test {TEST_NUM}")
	plt.grid(True)
	
	plot_filename = get_data_filepath(PLOTS_DIR, sensor_num)
	plt.savefig(plot_filename, dpi=300)
	plt.show()


# Running the analysis and plotting the results
for sensorNum in range(1, NUM_SENSORS + 1):
	arduino_filename = get_data_filepath(CALIBRATED_ARDUINO_DIR, sensorNum)
	excel_filename = get_data_filepath(ORIGINAL_INSTRON_DIR)
	
	arduino_time, arduino_force = read_arduino_data(arduino_filename, sensorNum)
	excel_time, excel_force = read_instron_data(excel_filename, sensorNum)
	
	plot_sensor_data(arduino_time, arduino_force, excel_time, excel_force, sensorNum)
