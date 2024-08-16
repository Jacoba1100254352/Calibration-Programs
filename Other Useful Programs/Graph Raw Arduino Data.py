import matplotlib.pyplot as plt

from Configuration_Variables import *


def read_uncalibrated_arduino_data(filename, sensor_num):
	"""
	Read and interpolate uncalibrated Arduino data from a text file.

	:param filename: Path, the file from which to read data.
	:param sensor_num: Integer, the number of the sensor.
	:return: Tuple, (arduino_time, arduino_force).
	"""
	with open(filename, "r") as file:
		lines = file.readlines()
	
	arduino_time = [(i + 1) * 20 for i in range(len(lines))]
	arduino_force = [float(line.split(",")[sensor_num].strip()) for line in lines]
	return arduino_time, arduino_force


def plot_sensor_data(time_data, force_data, sensor_num):
	"""
	Plot sensor data.

	:param time_data: Array-like, time data for the sensor.
	:param force_data: Array-like, force data for the sensor.
	:param sensor_num: Integer, the sensor number.
	"""
	plt.figure(figsize=(10, 6))
	plt.plot(time_data, force_data, label="Interpolated Uncalibrated Arduino Data", color="orange")
	
	plt.xlabel("Time [s]")
	plt.ylabel("Force [N]")
	plt.legend()
	plt.title(f"Comparison of Force Data for Sensor Set {SENSOR_SET}, Sensor {sensor_num}, Test {TEST_NUM}")
	plt.grid(True)
	plt.show()


# Running the analysis with interpolation and plotting the results
for sensorNum in range(1, NUM_SENSORS + 1):
	uncalibrated_filename = get_data_filepath(ORIGINAL_ARDUINO_DIR, sensorNum)
	
	arduino_time, arduino_force = read_uncalibrated_arduino_data(uncalibrated_filename, sensorNum)
	plot_sensor_data(arduino_time, arduino_force, sensorNum)
