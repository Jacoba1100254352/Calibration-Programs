# from matplotlib.backends.backend_pdf import PdfPages
import time

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error
from scipy.io import savemat

# from Configuration_Variables import *
# from Supplemental_Sensor_Graph_Functions import *
from Neural_Fit import *


# Set MATLAB-like appearance
# Define font sizes
SIZE_SMALL = 10
SIZE_DEFAULT = 14
SIZE_LARGE = 16
# plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

seed_value = 42  # Or any other integer


def analyze_and_graph_neural_fit(
	test_range, sensor_num, units=64, layers=2, activation='tanh', dropout_rate=0.5, l2_reg=0.01,
	learning_rate=0.001, epochs=100, batch_size=32, window_size=100, poly_order=None,
	smoothing_method="boxcar", save_graphs=True, show_graphs=True, bit_resolution=12,
	enable_hyperparameter_tuning=False, mapping='N_vs_N',
	hyperparams_dict=None
):
	plt.close('all')
	
	# Initialize the PDF to save the graphs
	with PdfPages(f"/Users/jacobanderson/Downloads/Neural_Network_Fit_Sensor_Set_{sensor_num}.pdf") as pdf:
		# Set up figures and axes for overlay and residuals
		overlay_fig, overlay_ax = plt.subplots(figsize=(10, 6))
		residuals_fig, residuals_ax = plt.subplots(figsize=(10, 6))
		
		for test_num in test_range:
			# Load and prepare data
			inputs, targets, instron_force, sensor_adc = load_and_prepare_data(
				sensor_num, test_num, bit_resolution, mapping
			)
			
			if enable_hyperparameter_tuning:
				# Train model with hyperparameter tuning
				model, input_scaler, output_scaler, best_hyperparams = train_model_with_hyperparameter_tuning(
					inputs, targets, bit_resolution, test_num, hyperparams_dict
				)
			else:
				# Train model without hyperparameter tuning
				model, input_scaler, output_scaler = train_model(
					inputs, targets, units, layers, activation, dropout_rate, l2_reg,
					learning_rate, epochs, batch_size, bit_resolution
				)
			
			# Evaluate model and calculate residuals
			outputs, residuals = evaluate_model(model, inputs, instron_force, sensor_adc, input_scaler, output_scaler, mapping)
			
			# Calculate MSE and MAE
			mse_nn = mean_squared_error(targets.flatten(), outputs.flatten())
			mae_nn = mean_absolute_error(targets.flatten(), outputs.flatten())
			
			# Print MSE and MAE for each test
			print(f"Test {test_num}, Neural Network Fit: MSE={mse_nn:.6f}, MAE={mae_nn:.6f}")
			
			# Apply smoothing to residuals
			residuals_smoothed = apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
			# targets = apply_smoothing(targets, method=smoothing_method, window_size=window_size, poly_order=poly_order)
			# outputs = apply_smoothing(outputs, method=smoothing_method, window_size=window_size, poly_order=poly_order)
			
			overlay_ax.set_xlim([0, -1])
			residuals_ax.set_xlim([0, -1])
			
			# First Graph: Plot calibrated sensor N vs Instron N
			overlay_ax.plot(targets.flatten(), outputs.flatten(), label=f"Calibrated Sensor [N] (Test {test_num})", linestyle='--', linewidth=2)
			overlay_ax.plot(targets.flatten(), targets.flatten(), label=f"Instron [N] (Test {test_num})", linewidth=2)
			
			# Customize the graph appearance
			overlay_ax.set_xlabel("Instron Force [N]")
			overlay_ax.set_ylabel("Calibrated Sensor Force [N]")
			overlay_ax.set_title(f"Neural Fit: Sensor vs. Instron Force ({bit_resolution}-bit)")  # Sensor-Instron Force Relationship with Neural Fit (12-bit)
			overlay_ax.legend(loc="lower left", fontsize=SIZE_SMALL, markerscale=0.8, labelspacing=0.3)
			overlay_ax.grid(True, which='both', linestyle='--', linewidth=0.75)  # Add grid lines
			# overlay_ax.invert_xaxis()
			
			# Plot residuals with MATLAB-style aesthetics
			residuals_ax.plot(instron_force.flatten(), residuals_smoothed, label=f"Residuals [N] (Test {test_num})", linewidth=2)
			residuals_ax.set_xlabel("Instron Force [N]")
			residuals_ax.set_ylabel("Residuals [N]")
			residuals_ax.set_title(f"Residuals: Sensor vs. Instron Force")
			residuals_ax.legend(loc="lower left", fontsize=SIZE_SMALL, markerscale=0.8, labelspacing=0.3)
			residuals_ax.grid(True, which='both', linestyle='--', linewidth=0.75)
			# residuals_ax.invert_xaxis()
		
		# Save and show graphs
		if save_graphs:
			pdf.savefig(overlay_fig)
			pdf.savefig(residuals_fig)
		
		if show_graphs:
			plt.show()
		
		plt.close(overlay_fig)
		plt.close(residuals_fig)


def analyze_and_graph_calibrated_data_and_fits_single_pdf_combined_multiple_tests(
	test_range, sensor_num, smoothing_method=None, window_size=100, poly_order=None,
	save_graphs=True, show_graphs=True, bit_resolution=12, mapping='N_vs_N'
):
	"""
	Analyze and visualize residuals and polynomial fits of different orders for each sensor across multiple tests,
	combining all tests in one graph per polynomial order.

	Parameters:
	- test_range: A range or list of test numbers to include in the analysis.
	- sensor_num: The sensor number to analyze.
	- window_size: Window size for the smoothing operation. If None, no smoothing is applied.
	- poly_order: Polynomial order for the Savitzky-Golay filter.
	- smoothing_method: The smoothing method ('savgol', 'boxcar', or None).
	- save_graphs: Boolean flag to save the graphs as a PDF.
	- show_graphs: Boolean flag to display the graphs during the process.
	- bit_resolution: The bit resolution to quantize the data (default is 12 bits).
	- mapping: Defines what to map on the x and y axes. Options are 'N_vs_N' (Calibrated N vs Instron N)
			   or 'ADC_vs_N' (ADC vs Instron N).
	"""
	# Replace SENSOR_SET and STARTING_SENSOR with appropriate variables or parameters if needed
	SENSOR_SET = "YourSensorSet"  # Update with your actual sensor set identifier
	STARTING_SENSOR = sensor_num
	
	with PdfPages(f"/Users/jacobanderson/Downloads/Combined_Tests_Polynomial_Sensor_Analysis_Sensor_Set_{SENSOR_SET}_Sensor_{STARTING_SENSOR}.pdf") as pdf:
		for order in range(1, 5):
			plt.figure(figsize=(10, 6))
			
			# Iterate over each test
			for _TEST_NUM in test_range:
				# Load data from CSV files
				instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
				updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
				
				# Ensure arrays are of equal length for accurate comparison
				min_length = min(len(instron_data), len(updated_arduino_data))
				instron_force = instron_data["Force [N]"].iloc[:min_length]
				
				# Depending on mapping, choose data for updated_arduino_force and set labels
				if mapping == 'N_vs_N':
					# Use calibrated force data
					calibrated_force_column = "Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"
					if calibrated_force_column in updated_arduino_data.columns:
						updated_arduino_force = updated_arduino_data[calibrated_force_column].iloc[:min_length]
						ylabel = "Residual Force [N]"
					else:
						print(f"Calibrated force data not found for sensor {sensor_num} in test {_TEST_NUM}.")
						continue  # Skip this test if calibrated data is missing
				elif mapping == 'ADC_vs_N':
					# Use raw ADC data
					adc_column = "ADC" if SIMPLIFY else f"ADC{sensor_num}"
					if adc_column in updated_arduino_data.columns:
						updated_arduino_force = updated_arduino_data[adc_column].iloc[:min_length]
						ylabel = "Residual Force [ADC]"
					else:
						print(f"ADC data not found for sensor {sensor_num} in test {_TEST_NUM}.")
						continue  # Skip this test if ADC data is missing
				else:
					raise ValueError("Invalid mapping type. Use 'N_vs_N' or 'ADC_vs_N'.")
				
				# Quantize input and output data
				instron_force = quantize_data(instron_force, bit_resolution)
				updated_arduino_force = quantize_data(updated_arduino_force, bit_resolution)
				
				# Fit the polynomial model
				lin_fit = calculate_line_of_best_fit(x=instron_force, y=updated_arduino_force, isPolyfit=True, order=order)
				residuals = updated_arduino_force - lin_fit
				
				# Calculate MSE and MAE for polynomial fit
				mse_poly = mean_squared_error(updated_arduino_force, lin_fit)
				mae_poly = mean_absolute_error(updated_arduino_force, lin_fit)
				
				print(f"Test {_TEST_NUM}, Polynomial Fit (Order {order}): MSE={mse_poly:.6f}, MAE={mae_poly:.6f}")
				
				# Apply smoothing using the specified method
				residuals_smoothed = apply_smoothing(
					residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order
				)
				
				# Plot the smoothed residuals
				plt.plot(instron_force[:len(residuals_smoothed)], residuals_smoothed, '-', label=f"Test {_TEST_NUM}", linewidth=2)
			
			plt.xlabel("Instron Force [N]")
			plt.ylabel(ylabel)
			plt.legend(loc="lower left")
			plt.title(f"Residuals for Polynomial Fit (Order {order}) Across Multiple Tests")
			plt.grid(True)
			plt.gca().invert_xaxis()
			
			if show_graphs:
				plt.show()
			
			if save_graphs:
				pdf.savefig()
			
			plt.close()


def analyze_and_graph_residuals_and_fits_individual_images(save_graphs=True, useArduinoADC=True):
	"""
	Analyze and export residuals and polynomial fits of different orders for each sensor into .mat files.
	"""
	for sensor_num in SENSORS_RANGE:
		# Sleep to avoid HTTPS request limit
		time.sleep(5)
		
		# Load data from CSV files
		instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
		updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num))
		
		# Extract time, force, and ADC data
		instron_time = instron_data["Time [s]"].to_numpy()
		instron_force = instron_data["Force [N]"].to_numpy()
		updated_arduino_time = updated_arduino_data["Time [s]"].to_numpy()
		updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"].to_numpy()
		
		# Get Aligned Arduino Data for ADC results to work regardless of SIMPLIFY's value
		aligned_arduino_data = pd.read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num))
		
		# Calculate force difference for export if needed
		difference = instron_force - updated_arduino_force
		
		# Create dictionary for .mat export
		data_dict = {
			'instron_time': instron_time,
			'instron_force': instron_force,
			'updated_arduino_time': updated_arduino_time,
			'updated_arduino_force': updated_arduino_force,
			'difference': difference
		}
		
		# Create filename for the .mat file (relevant to the sensor data)
		file_name = f"/Users/jacobanderson/Downloads/Test_{TEST_NUM}_Sensor_{sensor_num}_calibrated_forces.mat"
		
		# Save the data to a .mat file
		savemat(file_name, data_dict)
		
		print(f"Data for Sensor {sensor_num} saved to {file_name}")
		
		exit()
		
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
		plt.scatter(instron_force, arduino_force, label=f"Instron Force vs. Arduino {arduino_force_type}", color="purple")
		plt.xlabel("Instron Force [N]")
		plt.ylabel(f"Arduino {arduino_force_type} Values")
		plt.legend()
		plt.title(
			f"Relationship Between Instron Force and Arduino {arduino_force_type} Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
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
		plt.xlabel("Instron Force [N]")
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
		plt.xlabel("Instron Force [N]")
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
			
			plt.xlabel("Instron Force [N]")
			plt.ylabel(f"Arduino {arduino_force_type}")
			plt.legend()
			plt.title(f"Smoothed Residuals for Polynomial Fit of Order {order} - Sensor {sensor_num}")
			plt.grid(True)
			
			# Invert the x-axis
			plt.gca().invert_xaxis()
			
			if save_graphs:
				plt.savefig(f"/Users/jacobanderson/Downloads/Test {TEST_NUM} Sensor {sensor_num} order {order}.png", dpi=300)
			plt.show()
