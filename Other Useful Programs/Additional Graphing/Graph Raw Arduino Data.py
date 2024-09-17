import matplotlib.pyplot as plt

from Workflow_Programs.Configuration_Variables import *


def read_uncalibrated_arduino_data(filename, sensor_num, IRB_DATA):
	"""
	Read and interpolate uncalibrated Arduino data from a text file.

	:param filename: Path, the file from which to read data.
	:param sensor_num: Integer, the number of the sensor.
	:param IRB_DATA: Boolean, whether the data format is the IRB format.
	:return: Tuple, (arduino_time, arduino_force).
	"""
	with open(filename, "r") as file:
		lines = file.readlines()
	
	if IRB_DATA:
		# Assuming `lines` is the list of new formatted data as strings.
		arduino_time = [int(line.split()[0]) for line in lines]  # First value is the time in milliseconds
		
		arduino_force = []
		for line in lines:
			# Splitting the line into sections by whitespace, then grabbing the sensor values
			data_parts = line.split("Setpoint")[0].split()  # Split off everything after 'Setpoint'
			
			# The force values start after 'Pos: X'
			sensor_data = data_parts[11:11 + TOTAL_NUM_SENSORS]  # This skips "PID OFF", "Buz ON", "Pos: 0"
			# Parse sensor values (the next 9 entries in the line)
			force_values = [float(value.strip(',')) for value in sensor_data[:9]]
			arduino_force.append(force_values)
	else:
		arduino_time = [(i + 1) * 20 for i in range(len(lines))]
		arduino_force = [float(line.split(",")[sensor_num].strip()) for line in lines]
	
	return arduino_time, arduino_force


def plot_all_sensors(IRB_DATA, test_num):
	"""
	Plot sensor data for multiple sensors on the same plot, each with a unique color and label.

	:param time_data: Array-like, time data for the sensors.
	:param force_data_all: List of Array-like, force data for each sensor.
	:param test_num: Integer, the test number.
	:param total_num_sensors: Integer, total number of sensors.
	"""
	plt.figure(figsize=(10, 6))
	
	# Define a color map (optional, using a predefined color cycle in matplotlib)
	colors = plt.get_cmap("tab10")  # This gives a range of 10 different colors
	
	# Loop through all the sensors and plot each one with a different color and label
	for sensor_num in range(1, TOTAL_NUM_SENSORS + 1):
		arduino_time, arduino_force = read_uncalibrated_arduino_data(uncalibrated_filename, sensor_num, IRB_DATA)
		plt.plot(arduino_time, [inner_list[sensor_num - 1] for inner_list in arduino_force], label=f"Sensor {sensor_num}", color=colors(sensor_num - 1))
	
	# Add labels, title, legend, and grid
	plt.xlabel("Time [s]")
	plt.ylabel("Force [N]")
	plt.legend()
	plt.title(f"Comparison of Force Data for Sensor Set {SENSOR_SET}, Test {test_num}")
	plt.grid(True)
	plt.show()


IRB_DATA = True

path_adjustment = "../" if IRB_DATA else ""

# Reading the data for all sensors
all_force_data = []
uncalibrated_filename = path_adjustment + str(get_data_filepath(ORIGINAL_ARDUINO_DIR, "Test"))

# Plot all sensors on the same graph
plot_all_sensors(IRB_DATA, TEST_NUM)
