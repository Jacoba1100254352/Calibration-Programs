import time

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Workflow_Programs.Configuration_Variables import *
from Workflow_Programs.Supporting_and_Archived_Graphing_Functions.Supplemental_Sensor_Graph_Functions import *


def graph_sensor_data_difference():
	"""
    Generate and save plots comparing Instron and Arduino sensor data for each sensor,
    including the relationship between the Instron force and the Arduino ADC values.
    """
	for sensor_num in SENSORS_RANGE:
		# Load data from CSV files
		instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
		updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num))
		
		# Extract time, force, and ADC data
		instron_time, instron_force = instron_data["Time [s]"], instron_data["Force [N]"]
		updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
		
		# Plotting force comparison
		plt.figure(figsize=(10, 6))
		difference = instron_force - updated_arduino_force
		plt.plot(instron_time, difference, label="Difference (Instron - Updated Arduino)", color="green", linestyle="--")
		plt.xlabel("Time [s]")
		plt.ylabel("Force [N]")
		plt.legend()
		plt.title(f"Instron Arduino Force Difference for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
		plt.grid(True)
		
		plt.show()


def graph_sensor_data_difference_multi_test(test_range):
	"""
	Generate and display a single plot that compares the difference between Load Cell
	and Calibrated Sensor data across all specified tests and sensors.

	Parameters:
	test_range (iterable): A range or list of test numbers to include in the graph.

	The function reads data from CSV files, calculates the difference between Load Cell and Calibrated Sensor forces,
	and plots the differences on a single graph, with distinct colors for each sensor and test combination.
	"""
	
	# Initialize a single plot for all tests and sensors
	plt.figure(figsize=(12, 8))
	
	# Generate a unique color palette
	num_tests = len(test_range)
	num_sensors = len(SENSORS_RANGE)
	total_plots = num_tests * num_sensors
	colors = sns.color_palette("hsv", total_plots)
	
	color_index = 0
	
	# Loop over each test and sensor in the specified range
	for _TEST_NUM in test_range:
		for sensor_num in SENSORS_RANGE:
			# Load data from CSV files for the current sensor and test
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			
			# Extract relevant columns: time and force from Load Cell and Calibrated Sensor data
			instron_time, instron_force = instron_data["Time [s]"], instron_data["Force [N]"]
			updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
			
			# Calculate the difference between Load Cell and Calibrated Sensor forces
			difference = instron_force - updated_arduino_force
			
			# Plot the difference between Load Cell and Calibrated Sensor forces on the same plot
			plt.plot(instron_time, difference,
			         label=f"Test {_TEST_NUM}",
			         color=colors[color_index], linestyle="--")
			
			color_index += 1  # Increment color index for the next sensor/test combination
	
	# Finalize and display the combined plot
	plt.xlabel("Time [s]")
	plt.ylabel("Force Difference [N] (Load Cell - Sensor)")
	plt.legend()
	plt.title("Calibrated Sensor Force Deviation from Load Cell Baseline")
	plt.grid(True)
	plt.show()


def graph_sensor_average_best_fit():
	"""
	Generate and save plots comparing force measurements from the Load Cell and calibrated sensors
	for each sensor, including the calculated average force and its line of best fit.
	"""
	for sensor_num in SENSORS_RANGE:
		# Load data from CSV files
		instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
		updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num))
		
		# Extract time and force data
		instron_time, instron_force = instron_data["Time [s]"], instron_data["Force [N]"]
		updated_arduino_time = updated_arduino_data["Time [s]"]
		updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
		
		# Calculate the average force
		average_force = (instron_force + updated_arduino_force) / 2
		
		# Perform linear regression to find the line of best fit for average force
		line_of_best_fit_avg = calculate_line_of_best_fit(instron_time, average_force)
		
		# Plotting force comparison
		plt.figure(figsize=(10, 6))
		plt.plot(updated_arduino_time, updated_arduino_force, label="Calibrated Sensor Force", color="red")
		plt.plot(instron_time, instron_force, label="Reference Force (Load Cell)", color="blue")
		
		# Plotting lines of best fit
		plt.plot(instron_time, line_of_best_fit_avg, label="Average Force Best Fit", color="purple", linestyle="-.")
		
		plt.xlabel("Time [s]")
		plt.ylabel("Force [N]")
		plt.legend(loc='lower left')  # Set the legend location to the bottom left
		plt.title("Force Comparison and Average Best Fit: Calibrated Sensor vs. Baseline Load Cell")
		plt.grid(True)
		
		plt.show()


def graph_sensor_average_error(test_range):
	"""
	Generate and display a single plot that compares Instron and Arduino sensor data
	across all specified tests. The plot includes the relationship between the Instron force
	and the Arduino ADC values, adjusted by the combined average error for each sensor and test.

	Parameters:
	test_range (iterable): A range or list of test numbers to include in the graph.

	The function reads data from CSV files, calculates the combined average force,
	applies a line of best fit to the average force, and plots the errors for both
	the Instron and Arduino data on a single graph for easy comparison.
	"""
	
	# Initialize a single plot for all tests
	plt.figure(figsize=(12, 8))
	
	# Generate a unique color palette
	num_tests = len(test_range)
	num_sensors = len(SENSORS_RANGE)
	total_plots = num_tests * num_sensors * 2  # Multiplied by 2 for Instron and Arduino plots
	colors = sns.color_palette("hsv", total_plots)
	
	color_index = 0
	
	# Loop over each test and sensor in the specified range
	for _TEST_NUM in test_range:
		for sensor_num in SENSORS_RANGE:
			# Load data from CSV files for the current sensor and test
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			
			# Extract relevant columns: time and force from Instron and Arduino data
			instron_time, instron_force = instron_data["Time [s]"], instron_data["Force [N]"]
			updated_arduino_time = updated_arduino_data["Time [s]"]
			updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
			
			# Calculate the average force between Instron and Arduino data
			average_force = (instron_force + updated_arduino_force) / 2
			
			# Determine the line of best fit for the average force over time
			line_of_best_fit_avg = calculate_line_of_best_fit(instron_time, average_force)
			
			# Plot the error for Arduino data, adjusted by the average force line of best fit
			updated_arduino_force -= line_of_best_fit_avg
			plt.plot(updated_arduino_time, updated_arduino_force,
			         label=f"Updated Arduino Error (Sensor {sensor_num}, Test {_TEST_NUM})",
			         color=colors[color_index], alpha=0.7)
			
			# Plot the error for Instron data, adjusted by the average force line of best fit
			instron_force -= line_of_best_fit_avg
			plt.plot(instron_time, instron_force,
			         label=f"Instron Error (Sensor {sensor_num}, Test {_TEST_NUM})",
			         color=colors[color_index + 1], alpha=0.7)
			
			color_index += 2  # Increment color index for the next sensor/test combination
	
	# Finalize and display the plot
	plt.xlabel("Time [s]")
	plt.ylabel("Force [N]")
	plt.legend()
	plt.title(f"Combined Average Error Across Multiple Tests for {SENSOR_SET_DIR}")
	plt.grid(True)
	plt.show()


def graph_sensor_instron_error():
	"""
    Generate and save plots comparing Instron and Arduino sensor data for each sensor,
    including the relationship between the Instron force and the Arduino ADC values.
    """
	for sensor_num in SENSORS_RANGE:
		# Load data from CSV files
		instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
		
		# Extract time, force, and ADC data
		instron_time, instron_force = instron_data["Time [s]"], instron_data["Force [N]"]
		
		# Perform linear regression to find the line of best fit for instron force
		line_of_best_fit_instron = calculate_line_of_best_fit(instron_time, instron_force)
		
		# Plotting force comparison
		plt.figure(figsize=(10, 6))
		instron_force -= line_of_best_fit_instron
		plt.plot(instron_time, instron_force, label="Calibration Force Error", color="orange", linestyle=":")
		
		plt.xlabel("Time [s]")
		plt.ylabel("Force [N]")
		plt.legend()
		plt.title(f"Instron Error for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
		plt.grid(True)
		
		plt.show()


def graph_sensor_arduino_error():
	"""
    Generate and save plots comparing Instron and Arduino sensor data for each sensor,
    including the relationship between the Instron force and the Arduino ADC values.
    """
	for sensor_num in SENSORS_RANGE:
		# Load data from CSV files
		updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num))
		
		# Extract time, force, and ADC data
		updated_arduino_time = updated_arduino_data["Time [s]"]
		updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
		
		# Perform linear regression to find the line of best fit for instron force
		line_of_best_fit_arduino = calculate_line_of_best_fit(updated_arduino_time, updated_arduino_force)
		
		# Plotting force comparison # Plotting lines of best fit
		plt.figure(figsize=(10, 6))
		updated_arduino_force -= line_of_best_fit_arduino
		plt.plot(updated_arduino_time, updated_arduino_force, label="Arduino Force Error", color="orange", linestyle=":")
		
		plt.xlabel("Time [s]")
		plt.ylabel("Force [N]")
		plt.legend()
		plt.title(f"Arduino Error for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
		plt.grid(True)
		
		plt.show()


def analyze_and_graph_residuals_and_fits_individual_images(save_graphs=True, useArduinoADC=True):
	"""
    Analyze and visualize residuals and polynomial fits of different orders for each sensor.
    """
	for sensor_num in SENSORS_RANGE:
		# Sleep to avoid HTTPS request limit
		time.sleep(5)
		
		# Load data from CSV files
		instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
		updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num))
		
		# Extract time, force, and ADC data
		instron_time, instron_force = instron_data["Time [s]"], instron_data["Force [N]"]
		updated_arduino_time = updated_arduino_data["Time [s]"]
		updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
		
		# Get Aligned Arduino Data for ADC results to work regardless of SIMPLIFY's value
		aligned_arduino_data = pd.read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num))
		
		# Plotting force comparison
		plt.figure(figsize=(10, 6))
		plt.plot(updated_arduino_time, updated_arduino_force, label="Updated Arduino Data", color="red")
		plt.plot(instron_time, instron_force, label="Instron Data", color="blue")
		difference = instron_force - updated_arduino_force
		plt.plot(instron_time, difference, label="Difference (Instron - Updated Arduino)", color="green", linestyle="--")
		plt.xlabel("Time [s]")
		plt.ylabel("Force [N]")
		plt.legend()
		plt.title(f"Comparison of Force Data for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
		plt.grid(True)
		if save_graphs:
			plt.savefig(f"/Users/jacobanderson/Downloads/Test {TEST_NUM} Sensor {sensor_num} calibrated forces.png", dpi=300)
		plt.show()
		
		# Ensure arrays are of equal length for accurate comparison
		min_length = min(len(instron_data), len(updated_arduino_data))
		instron_force = instron_data["Force [N]"].iloc[:min_length]
		if useArduinoADC:
			arduino_force_type = "ADC" if SIMPLIFY else f"ADC{sensor_num}"
			arduino_force = aligned_arduino_data["ADC" if SIMPLIFY else f"ADC{sensor_num}"].iloc[:min_length]
		else:
			arduino_force_type = "Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"
			arduino_force = updated_arduino_force.iloc[:min_length]
		
		# Second plot: Relationship between Instron force and Arduino ADC values
		plt.figure(figsize=(10, 6))
		plt.scatter(instron_force, arduino_force, label=f"Calibration Force vs. Arduino {arduino_force_type}", color="purple")
		plt.xlabel("Calibration Force [N]")
		plt.ylabel(f"Arduino {arduino_force_type} Values")
		plt.legend()
		plt.title(
			f"Relationship Between Calibration Force and Arduino {arduino_force_type} Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
		plt.grid(True)
		
		# Invert the x-axis
		plt.gca().invert_xaxis()
		
		if save_graphs:
			plt.savefig(f"/Users/jacobanderson/Downloads/Test {TEST_NUM} Sensor {sensor_num} adc against N.png", dpi=300)
		plt.show()
		
		# Calculate and plot the best-fit line
		lin_fit = calculate_line_of_best_fit(instron_force, arduino_force)
		
		# Plot the best-fit line over the scatter plot
		plt.figure(figsize=(10, 6))
		plt.scatter(instron_force, arduino_force, label="Actual Data", color="purple")
		plt.plot(instron_force, lin_fit, label="Best-fit line", color="orange")
		plt.xlabel("Calibration Force [N]")
		plt.ylabel(f"Arduino {arduino_force_type} Values")
		plt.legend()
		plt.title(
			f"Best-fit Line Through {arduino_force_type} Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
		plt.grid(True)
		
		# Invert the x-axis
		plt.gca().invert_xaxis()
		
		if save_graphs:
			plt.savefig(f"/Users/jacobanderson/Downloads/Test {TEST_NUM} Sensor {sensor_num} best-fit line through {arduino_force_type} values.png", dpi=300)
		plt.show()
		
		# Calculate and plot residuals
		residuals = arduino_force - lin_fit
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
		plt.xlabel("Calibration Force [N]")
		plt.ylabel("Residuals")
		plt.legend()
		plt.title(
			f"Smoothed Residuals of {arduino_force_type} Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
		plt.grid(True)
		
		# Invert the x-axis
		plt.gca().invert_xaxis()
		
		if save_graphs:
			plt.savefig(f"/Users/jacobanderson/Downloads/Test {TEST_NUM} Sensor {sensor_num} averaged noise.png", dpi=300)
		plt.show()
		
		# Fit and plot polynomial models of 1st through 4th order
		for order in range(1, 5):
			# Sleep to avoid HTTPS request limit
			time.sleep(2)
			
			# Fit the polynomial model
			coefficients = np.polyfit(instron_force, arduino_force, order)
			polynomial = np.poly1d(coefficients)
			predicted_adc_force = polynomial(instron_force)
			residuals = arduino_force - predicted_adc_force
			
			# Smooth the residuals
			residuals_smoothed = np.convolve(residuals, np.ones(window_size) / window_size, mode='valid')
			instron_force_adjusted = instron_force[(window_size - 1):]  # Adjusting the x-axis for smoothed residuals
			
			plt.figure(figsize=(10, 6))
			plt.plot(instron_force_adjusted, residuals_smoothed, '-', label=f"Order {order} smoothed residuals",
			         linewidth=2)
			
			if order == 1:
				# For first-order, you might still want to plot the average line for comparison
				average_residual = np.mean(residuals)
				plt.axhline(y=average_residual, color='r', linestyle='-', label="Average Residual")
			
			plt.xlabel("Calibration Force [N]")
			plt.ylabel(f"Arduino {arduino_force_type}")
			plt.legend()
			plt.title(f"Smoothed Residuals for Polynomial Fit of Order {order} - Sensor {sensor_num}")
			plt.grid(True)
			
			# Invert the x-axis
			plt.gca().invert_xaxis()
			
			if save_graphs:
				plt.savefig(f"/Users/jacobanderson/Downloads/Test {TEST_NUM} Sensor {sensor_num} order {order}.png", dpi=300)
			plt.show()


def analyze_and_graph_residuals_and_fits_single_pdf(test_range):
	"""
    Analyze and visualize residuals and polynomial fits of different orders for each sensor.
    """
	# Initialize a PDF file to save all the plots
	for _TEST_NUM in test_range:
		with PdfPages(f"/Users/jacobanderson/Downloads/Test_{_TEST_NUM}_Sensor_2_Sensor_Analysis.pdf") as pdf:
			for sensor_num in SENSORS_RANGE:
				# Sleep to avoid HTTPS request limit
				time.sleep(5)
				
				# Load data from CSV files
				instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
				updated_arduino_data = pd.read_csv(
					get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
				
				# Extract time, force, and ADC data
				instron_time, instron_force = instron_data["Time [s]"], instron_data["Force [N]"]
				updated_arduino_time = updated_arduino_data["Time [s]"]
				updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
				
				# Get Aligned Arduino Data for ADC results to work regardless of SIMPLIFY's value
				aligned_arduino_data = pd.read_csv(
					get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
				
				# Plotting force comparison
				plt.figure(figsize=(10, 6))
				plt.plot(updated_arduino_time, updated_arduino_force, label="Updated Arduino Data", color="red")
				plt.plot(instron_time, instron_force, label="Instron Data", color="blue")
				difference = instron_force - updated_arduino_force
				plt.plot(instron_time, difference, label="Difference (Instron - Updated Arduino)", color="green",
				         linestyle="--")
				plt.xlabel("Time [s]")
				plt.ylabel("Force [N]")
				plt.legend()
				plt.title(f"Comparison of Force Data for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {_TEST_NUM}")
				plt.grid(True)
				pdf.savefig()  # Save the current figure to PDF
				plt.close()
				
				# Ensure arrays are of equal length for accurate comparison
				min_length = min(len(instron_data), len(updated_arduino_data))
				instron_force = instron_data["Force [N]"].iloc[:min_length]
				updated_arduino_adc_force = aligned_arduino_data["ADC" if SIMPLIFY else f"ADC{sensor_num}"].iloc[
				                            :min_length]
				
				# Second plot: Relationship between Instron force and Arduino ADC values
				plt.figure(figsize=(10, 6))
				plt.scatter(instron_force, updated_arduino_adc_force, label="Calibration Force vs. Arduino ADC",
				            color="purple")
				plt.xlabel("Calibration Force [N]")
				plt.ylabel(f"Arduino ADC{sensor_num} Values")
				plt.legend()
				plt.title(
					f"Relationship Between Calibration Force and Arduino ADC{sensor_num} Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {_TEST_NUM}")
				plt.grid(True)
				
				# Invert the x-axis
				plt.gca().invert_xaxis()
				
				pdf.savefig()
				plt.close()
				
				# Calculate and plot the best-fit line
				coefficients = np.polyfit(instron_force, updated_arduino_adc_force, 1)
				polynomial = np.poly1d(coefficients)
				lin_fit = polynomial(instron_force)
				
				# Plot the best-fit line over the scatter plot
				plt.figure(figsize=(10, 6))
				plt.scatter(instron_force, updated_arduino_adc_force, label="Actual Data", color="purple")
				plt.plot(instron_force, lin_fit, label="Best-fit line", color="orange")
				plt.xlabel("Calibration Force [N]")
				plt.ylabel(f"Arduino ADC{sensor_num} Values")
				plt.legend()
				plt.title(
					f"Best-fit Line Through ADC{sensor_num} Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {_TEST_NUM}")
				plt.grid(True)
				
				# Invert the x-axis
				plt.gca().invert_xaxis()
				
				pdf.savefig()
				plt.close()
				
				# Calculate and plot residuals
				residuals = updated_arduino_adc_force - lin_fit
				print("Length of residuals:", len(residuals))
				plt.figure(figsize=(10, 6))
				plt.scatter(instron_force, residuals, label="Residuals", color="green")
				
				# Calculate a simple moving average of the residuals
				window_size = 1000  # Choose a window size that makes sense for your data
				residuals_smoothed = np.convolve(residuals, np.ones(window_size) / window_size, mode='valid')
				
				# To plot the smoothed residuals, we need to adjust the x-axis (instron_force) due to the convolution operation
				# This adjustment depends on the 'mode' used in np.convolve. With 'valid', the length of the output is N - K + 1
				instron_force_adjusted = instron_force[(window_size - 1):]  # Adjusting the x-axis
				
				plt.plot(instron_force_adjusted, residuals_smoothed, label="Smoothed Residuals", color="blue",
				         linewidth=2)
				plt.axhline(y=0, color='r', linestyle='-')
				plt.xlabel("Calibration Force [N]")
				plt.ylabel("Residuals")
				plt.legend()
				plt.title(
					f"Smoothed Residuals of ADC{sensor_num} Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {_TEST_NUM}")
				plt.grid(True)
				
				# Invert the x-axis
				plt.gca().invert_xaxis()
				
				pdf.savefig()
				plt.close()
				
				# Fit and plot polynomial models of 1st through 4th order
				for order in range(1, 5):
					# Sleep to avoid HTTPS request limit
					time.sleep(2)
					
					# Fit the polynomial model
					coefficients = np.polyfit(instron_force, updated_arduino_adc_force, order)
					polynomial = np.poly1d(coefficients)
					predicted_adc_force = polynomial(instron_force)
					residuals = updated_arduino_adc_force - predicted_adc_force
					
					# Smooth the residuals
					residuals_smoothed = np.convolve(residuals, np.ones(window_size) / window_size, mode='valid')
					instron_force_adjusted = instron_force[
					                         (window_size - 1):]  # Adjusting the x-axis for smoothed residuals
					
					plt.figure(figsize=(10, 6))
					plt.plot(instron_force_adjusted, residuals_smoothed, '-', label=f"Order {order} smoothed residuals",
					         linewidth=2)
					
					if order == 1:
						# For first-order, you might still want to plot the average line for comparison
						average_residual = np.mean(residuals)
						plt.axhline(y=average_residual, color='r', linestyle='-', label="Average Residual")
					
					plt.xlabel("Calibration Force [N]")
					plt.ylabel("Residuals")
					plt.legend()
					plt.title(f"Smoothed Residuals for Polynomial Fit of Order {order} - Sensor {sensor_num}")
					plt.grid(True)
					
					# Invert the x-axis
					plt.gca().invert_xaxis()
					
					pdf.savefig()
					plt.close()


def analyze_and_graph_residuals_and_fits_single_pdf_combined_multiple_tests(
	test_range, save_graphs=True, show_graphs=True, smoothing_method="boxcar", window_size=100, poly_order=None
):
	"""
    Analyze and visualize residuals and polynomial fits of different orders for each sensor across multiple tests,
    combining all tests in one graph per polynomial order.
    """
	# Iterate over polynomial orders
	with PdfPages(f"/Users/jacobanderson/Downloads/Combined_Tests_Polynomial_Sensor_Analysis_Sensor_Set_{SENSOR_SET}_Sensor_{STARTING_SENSOR}.pdf") as pdf:
		for order in range(1, 5):
			plt.figure(figsize=(10, 6))
			
			# Iterate over each test
			for _TEST_NUM in test_range:
				for sensor_num in SENSORS_RANGE:
					# Load data from CSV files
					instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
					updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
					aligned_arduino_data = pd.read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
					
					# Ensure arrays are of equal length for accurate comparison
					min_length = min(len(instron_data), len(updated_arduino_data))
					instron_force = instron_data["Force [N]"].iloc[:min_length]
					updated_arduino_adc_force = aligned_arduino_data["ADC" if SIMPLIFY else f"ADC{sensor_num}"].iloc[:min_length]
					
					# Fit the polynomial model
					lin_fit = calculate_line_of_best_fit(instron_force, updated_arduino_adc_force, isPolyfit=True, order=order)
					residuals = updated_arduino_adc_force - lin_fit
					
					# Smooth the residuals
					residuals_smoothed = apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
					instron_force_adjusted = instron_force[:len(residuals_smoothed)]
					
					plt.plot(instron_force_adjusted, residuals_smoothed, '-', label=f"Test {_TEST_NUM}, Sensor {sensor_num}", linewidth=2)
			
			plt.xlabel("Calibration Force [N]")
			plt.ylabel("Residuals")
			plt.legend()
			plt.title(f"Smoothed Residuals for Polynomial Fit of Order {order} - Combined Tests")
			plt.grid(True)
			plt.gca().invert_xaxis()
			
			if show_graphs:
				plt.show()
			
			if save_graphs:
				pdf.savefig()
			
			plt.close()


def plot_Instron_force_vs_Arduino_force(test_num, sensor_num, show_graphs=True, show_polyfit=True, order=1):
	"""
    Plot the relationship between Instron force and Arduino force for a specific sensor and test.
    """
	
	# Load data from CSV files
	instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=test_num))
	updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=test_num))
	
	# Ensure arrays are of equal length for accurate comparison
	min_length = min(len(instron_data), len(updated_arduino_data))
	instron_force = instron_data["Force [N]"].iloc[:min_length]
	updated_arduino_force = arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"].iloc[:min_length]
	
	plt.figure(figsize=(10, 6))
	
	plt.scatter(instron_force, updated_arduino_force, label="Calibration Force [N] vs. Arduino Force [N]", color="purple")
	
	plt.xlabel("Calibration Force [N]")
	plt.ylabel("Calibrated Arduino Force [N]")
	plt.legend()
	plt.title(f"Force [N] Instron vs. Arduino for Sensor {sensor_num}, Test {test_num}")
	plt.grid(True)
	plt.gca().invert_xaxis()
	
	if show_polyfit:
		# Calculate and plot the best-fit line
		lin_fit = calculate_line_of_best_fit(instron_force, arduino_force, isPolyfit=True, order=order)
		
		# Plot the best-fit line over the scatter plot
		plt.plot(instron_force, lin_fit, label="Best-fit line", color="orange")
		plt.xlabel("Calibration Force [N]")
		plt.ylabel(f"Arduino Force [N] Values")
		plt.legend()
		plt.title(
			f"Best-fit Line Through Force [N] Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
		plt.grid(True)
		
		# Invert the x-axis
		plt.gca().invert_xaxis()
	
	if show_graphs:
		plt.show()
	
	plt.close()


def plot_adjusted_linear_fits(test_range):
	"""
    Plot and save the adjusted linear fits for different tests on the same graph.
    This adjustment involves normalizing the slopes around the average slope of all fits
    and plotting these adjustments to focus on slight slope variations and offsets.
    """
	plt.figure(figsize=(10, 6))
	slopes = []
	
	# First, calculate the average slope across all tests and sensors
	for _TEST_NUM in test_range:
		for sensor_num in SENSORS_RANGE:
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			
			min_length = min(len(instron_data), len(updated_arduino_data))
			instron_force = instron_data["Force [N]"].iloc[:min_length]
			updated_arduino_adc_force = updated_arduino_data["ADC" if SIMPLIFY else f"ADC{sensor_num}"].iloc[:min_length]
			
			coefficients = np.polyfit(instron_force, updated_arduino_adc_force, 1)
			slopes.append(coefficients[0])
	
	average_slope = np.mean(slopes)
	
	# Then, adjust the slope of each fit by subtracting the average slope and plot the results
	for _TEST_NUM in test_range:
		for sensor_num in SENSORS_RANGE:
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			
			min_length = min(len(instron_data), len(updated_arduino_data))
			instron_force = instron_data["Force [N]"].iloc[:min_length]
			updated_arduino_adc_force = updated_arduino_data["ADC" if SIMPLIFY else f"ADC{sensor_num}"].iloc[:min_length]
			
			coefficients = np.polyfit(instron_force, updated_arduino_adc_force, 1)
			adjusted_slope = coefficients[0] - average_slope
			adjusted_intercept = coefficients[1]
			
			# Create a polynomial with the adjusted slope and the original intercept
			adjusted_polynomial = np.poly1d([adjusted_slope, adjusted_intercept])
			adjusted_fit = adjusted_polynomial(instron_force)
			
			plt.plot(instron_force, adjusted_fit, label=f"Test {_TEST_NUM}, Sensor {sensor_num}")
	
	plt.xlabel("Calibration Force [N]")
	plt.ylabel("Adjusted ADC Values")
	plt.legend()
	plt.title("Adjusted Linear Fits with Normalized Slopes Across Tests")
	plt.grid(True)
	
	# Invert the x-axis if that matches your previous plots
	plt.gca().invert_xaxis()
	
	# Save the plot
	plt.savefig("/Users/jacobanderson/Downloads/adjusted_linear_fits_across_tests.png")  # Update the path as needed
	plt.close()


def plot_adjusted_linear_fits_no_offsets(test_range, save_graphs=True, useArduinoADC=True):
	"""
	This function plots adjusted linear fits of force data from an Instron machine and Arduino sensors,
	centered at zero, across multiple tests and sensors. The adjustments are based on the average slope
	calculated across all test cases and sensors.

	Parameters:
	- test_range: A list or range object containing the test numbers to process.
	- save_graphs: A boolean indicating whether to save the graph as a PNG file. If False, the graph is displayed instead.
	- useArduinoADC: A boolean indicating whether to use Arduino ADC values (True) or force values (False).

	The function outputs either a saved image file or a displayed graph, depending on the save_graphs parameter.
	"""
	
	# Initialize plotting and a list to store slopes
	plt.figure(figsize=(10, 6))
	slopes = []
	
	# First loop: Calculate the average slope across all tests and sensors
	for test_num in test_range:
		for sensor_num in SENSORS_RANGE:
			# Load and truncate data for the current test number and sensor
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=test_num))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=test_num))
			aligned_arduino_data = pd.read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num))
			
			min_length = min(len(instron_data), len(updated_arduino_data))
			instron_force = instron_data["Force [N]"].iloc[:min_length]
			
			# Decide which Arduino data to use: ADC or force
			if useArduinoADC:
				arduino_force = aligned_arduino_data["ADC" if SIMPLIFY else f"ADC{sensor_num}"].iloc[:min_length]
			else:  # Updated contains the new/calibrated values at the time of calibration (new CSV)
				arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"].iloc[:min_length]
			
			# Perform linear fitting and store the slope
			coefficients = np.polyfit(instron_force, arduino_force, 1)
			slopes.append(coefficients[0])
	
	# Calculate the average slope from all slopes collected
	average_slope = np.mean(slopes)
	arduino_force_type = "default"
	
	# Second loop: Adjust and plot the data
	for test_num in test_range:
		for sensor_num in SENSORS_RANGE:
			# Reload and truncate data for the current test and sensor
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=test_num))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=test_num))
			aligned_arduino_data = pd.read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num))
			
			# Ensure arrays are of equal length for accurate comparison
			min_length = min(len(instron_data), len(updated_arduino_data))
			instron_force = instron_data["Force [N]"].iloc[:min_length]
			
			# Decide which Arduino data to use: ADC or force
			if useArduinoADC:
				arduino_force_type = "ADC" if SIMPLIFY else f"ADC{sensor_num}"
				arduino_force = aligned_arduino_data[arduino_force_type].iloc[:min_length]
			else:
				arduino_force_type = "Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"
				arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"].iloc[:min_length]
			
			# Perform linear fitting, adjust the slope, and compute adjusted polynomial
			coefficients = np.polyfit(instron_force, arduino_force, 1)
			adjusted_slope = coefficients[0] - average_slope
			adjusted_polynomial = np.poly1d([adjusted_slope, coefficients[1]])
			adjusted_fit = adjusted_polynomial(instron_force)
			
			# Center the adjusted fit at zero and plot it
			adjusted_fit_centered = adjusted_fit - np.mean(adjusted_fit)
			plt.plot(instron_force, adjusted_fit_centered, label=f"Test {test_num}, Sensor {sensor_num}")
	
	# Finalize and display or save the plot
	arduino_force_type = "ADC" if useArduinoADC else "N"
	plt.xlabel("Measured Force by Instron [N]")
	plt.ylabel(f"Zero-Centered Adjusted Arduino {arduino_force_type}")
	plt.legend()
	plt.title("Zero-Centered Adjusted Sensor Data Across Tests")
	plt.grid(True)
	plt.gca().invert_xaxis()
	
	if save_graphs:
		plt.savefig(f"/Users/jacobanderson/Downloads/zero_centered_adjusted_sensor_fits_Tests_{test_range[0]}-{test_range[-1]}_Sensor_{SENSORS_RANGE[0]}_{arduino_force_type}_N.png")
	else:
		plt.show()
	
	plt.close()


def graph_and_verify_calibration():
	"""
    Compare new data with previous calibrations against the new/relevant instron data.
    """
	
	if NUM_SENSORS != 1:
		raise ValueError("This function is only intended for a single sensor.")
	
	sensor_num = STARTING_SENSOR
	
	# Load data from CSV files
	instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
	aligned_arduino_data = pd.read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num))
	
	# Extract time, force, and ADC data
	instron_time, instron_force = instron_data["Time [s]"], instron_data["Force [N]"]
	aligned_arduino_time = aligned_arduino_data["Time [s]"]
	aligned_arduino_force = -aligned_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
	print(instron_time)
	
	# Plotting force comparison
	plt.figure(figsize=(10, 6))
	difference = instron_force - aligned_arduino_force
	aligned_arduino_force += difference_polarity(instron_force, aligned_arduino_force) * avg(difference)
	difference = instron_force - aligned_arduino_force
	
	plt.plot(instron_time, difference, label="Difference (Instron - Updated Arduino)", color="green", linestyle="--")
	plt.xlabel("Time [s]")
	plt.ylabel("Force [N]")
	plt.legend()
	plt.title(f"Previous Calibration and Calibration Force Difference for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
	plt.grid(True)
	
	plt.show()
	
	# Plotting force comparison
	plt.figure(figsize=(10, 6))
	plt.plot(instron_time, difference, label="Difference (Instron - Updated Arduino)", color="green", linestyle="--")
	aligned_arduino_force += avg(difference)
	plt.plot(aligned_arduino_time, aligned_arduino_force, label="Updated Arduino Data", color="red")
	plt.plot(instron_time, instron_force, label="Instron Data", color="blue")
	plt.xlabel("Time [s]")
	plt.ylabel("Force [N]")
	plt.legend()
	plt.title(f"Comparison of Force Data for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
	plt.grid(True)
	
	plt.show()
