import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Configuration_Variables import *


for sensor_num in SENSORS_RANGE:
	# Load data from CSV files
	instron_data = pd.read_csv(get_data_filepath(PARSED_INSTRON_DIR, sensor_num))
	aligned_arduino_data = pd.read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num))
	
	# Ensure arrays are of equal length for accurate comparison
	min_length = min(len(instron_data), len(aligned_arduino_data))
	instron_force = instron_data["Force [N]"].iloc[:min_length]
	aligned_arduino_force = aligned_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"].iloc[:min_length]
	
	# Second plot: Relationship between Instron force and Arduino ADC values
	plt.figure(figsize=(10, 6))
	plt.scatter(instron_force, aligned_arduino_force, label="Instron Force vs. Arduino Force [N]",
	            color="purple")
	plt.xlabel("Instron Force [N]")
	plt.ylabel(f"Arduino Force{sensor_num} [N] Values")
	plt.legend()
	plt.title(
		f"Relationship Between Instron Force and Arduino Force{sensor_num} [N] Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
	plt.grid(True)
	
	# Invert the x-axis
	plt.gca().invert_xaxis()
	
	plt.show()
	
	# Calculate and plot the best-fit line
	coefficients = np.polyfit(instron_force, aligned_arduino_force, 1)
	polynomial = np.poly1d(coefficients)
	lin_fit = polynomial(instron_force)
	
	# Plot the best-fit line over the scatter plot
	plt.figure(figsize=(10, 6))
	plt.scatter(instron_force, aligned_arduino_force, label="Actual Data", color="purple")
	plt.plot(instron_force, lin_fit, label="Best-fit line", color="orange")
	plt.xlabel("Instron Force [N]")
	plt.ylabel(f"Arduino Force{sensor_num} [N]")
	plt.legend()
	plt.title(
		f"Best-fit Line Through Force{sensor_num} [N] Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
	plt.grid(True)
	
	# Invert the x-axis
	plt.gca().invert_xaxis()
	
	plt.show()
	
	# Calculate and plot residuals
	residuals = aligned_arduino_force - lin_fit
	print("Length of residuals:", len(residuals))
	plt.figure(figsize=(10, 6))
	plt.scatter(instron_force, residuals, label="Residuals", color="green")
	
	# Calculate a simple moving average of the residuals
	window_size = 1000  # Choose a window size that makes sense for your data
	residuals_smoothed = np.convolve(residuals, np.ones(window_size) / window_size, mode='valid')
	
	# To plot the smoothed residuals, we need to adjust the x-axis (instron_force) due to the convolution operation
	# This adjustment depends on the 'mode' used in np.convolve. With 'valid', the length of the output is N - K + 1
	instron_force_adjusted = instron_force[(window_size - 1):]  # Adjusting the x-axis
	
	plt.plot(instron_force_adjusted, residuals_smoothed, label="Smoothed Residuals", color="blue", linewidth=2)
	plt.axhline(y=0, color='r', linestyle='-')
	plt.xlabel("Instron Force [N]")
	plt.ylabel("Residuals")
	plt.legend()
	plt.title(f"Smoothed Residuals of Force{sensor_num} [N] Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
	plt.grid(True)
	
	# Invert the x-axis
	plt.gca().invert_xaxis()
	
	plt.show()
