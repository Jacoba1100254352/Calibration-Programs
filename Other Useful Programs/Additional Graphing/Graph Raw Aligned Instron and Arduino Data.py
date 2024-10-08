import matplotlib.pyplot as plt
import pandas as pd

from Workflow_Programs.Configuration_Variables import *
from Workflow_Programs.Supplemental_Sensor_Graph_Functions import apply_smoothing


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


def plot_sensor_data(arduino_time, arduino_force, instron_time, instron_force, sensor_num):
	"""
	Plot and display the sensor data.

	:param arduino_time: Array-like, time data from Arduino.
	:param arduino_force: Array-like, force data from Arduino.
	:param instron_time: Array-like, time data from Instron.
	:param instron_force: Array-like, force data from Instron.
	:param sensor_num: Integer, the sensor number.
	"""
	fig, ax1 = plt.subplots(figsize=(10, 6))
	ax1.plot(arduino_time, arduino_force, label=f"Uncalibrated Sensor [ADC] (Test {TEST_NUM})", color="red")
	ax1.set_xlabel("Time [s]")
	ax1.set_ylabel("Uncalibrated Sensor [ADC]", color="red")
	ax1.tick_params(axis="y", labelcolor="red")
	
	ax2 = ax1.twinx()
	ax2.plot(instron_time, instron_force, label=f"Instron [N] (Test {TEST_NUM})", color="blue")
	ax2.set_ylabel(f"Instron Force [N]", color="blue")
	ax2.tick_params(axis="y", labelcolor="blue")
	
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2, loc=0)
	
	plt.title(f"Overlay: Uncalibrated Sensor vs Instron Force")
	plt.grid(True)
	plt.show()


# Running the analysis and plotting the results
for sensor_num in range(2, 3):
	arduino_filename = "../" + str(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num))
	instron_filename = "../" + str(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
	
	arduino_time, arduino_force = pd.read_csv(arduino_filename)["Time [s]"], pd.read_csv(arduino_filename)[
		f"ADC" if SIMPLIFY else f"ADC{sensor_num}"]  #read_sensor_data(arduino_filename)
	instron_time, instron_force = read_sensor_data(instron_filename)
	
	# ins