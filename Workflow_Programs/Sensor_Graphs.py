# from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error

# from Configuration_Variables import *
# from Supplemental_Sensor_Graph_Functions import *
from Neural_Fit import *


def analyze_and_graph_neural_fit(
	test_range, sensor_num, units=64, layers=2, activation='relu', dropout_rate=0.5, l2_reg=0.01,
	learning_rate=0.001, epochs=100, batch_size=32, window_size=None, poly_order=None,
	smoothing_method="boxcar", save_graphs=True, show_graphs=True, bit_resolution=12,
	enable_hyperparameter_tuning=False, mapping='ADC_vs_N',
	hyperparams_dict=None
):
	"""
	Analyze and visualize neural network fits for each test and compare them across multiple tests.
	This function fits a neural network model to each test individually, graphs the combined data and neural fit,
	subtracts the individual neural fit from the combined data, and graphs the residuals for each test.
	"""
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
			
			# Apply smoothing to residuals
			residuals_smoothed = apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
			
			# Plot overlay
			plot_overlay(overlay_ax, inputs, targets, outputs, test_num, mapping)
			
			# Plot residuals
			plot_residuals(residuals_ax, inputs, residuals_smoothed, test_num, mapping)
		
		# Finalize and save plots
		overlay_ax.set_title(f"Overlay of Data and Neural Fit for Tests {test_range}")
		overlay_ax.legend(loc="upper left")
		if save_graphs:
			pdf.savefig(overlay_fig)
		
		residuals_ax.set_title(f"Residuals for Tests {test_range}")
		residuals_ax.legend(loc="upper left")
		if save_graphs:
			pdf.savefig(residuals_fig)
		
		if show_graphs:
			plt.show()
		plt.close(overlay_fig)
		plt.close(residuals_fig)


def analyze_and_graph_calibrated_data_and_fits_single_pdf_combined_multiple_tests(
	test_range, sensor_num, window_size=None, poly_order=None, smoothing_method=None, save_graphs=True,
	show_graphs=True, bit_resolution=12, segment_size=None
):
	"""
	Analyze and visualize residuals and polynomial fits of different orders for each sensor across multiple tests,
	combining all tests in one graph per polynomial order. Optionally, you can segment the data (similar to batch processing).

	Parameters:
	- test_range: A range or list of test numbers to include in the analysis.
	- sensor_num: The sensor number to analyze.
	- window_size: Window size for the smoothing operation. If None, no smoothing is applied.
	- poly_order: Polynomial order for the polynomial fit.
	- smoothing_method: The smoothing method ('savgol', 'boxcar', or None).
	- save_graphs: Boolean flag to save the graphs as a PDF.
	- show_graphs: Boolean flag to display the graphs during the process.
	- bit_resolution: The bit resolution to quantize the data (default is 12 bits).
	- segment_size: If provided, splits data into smaller segments for polynomial fitting (mimicking batch processing).
	"""
	
	# Create the PDF to save the graphs
	with PdfPages(f"/Users/jacobanderson/Downloads/Combined_Tests_Polynomial_Sensor_Analysis_Sensor_Set_{sensor_num}.pdf") as pdf:
		for order in range(1, 5):  # Loop through polynomial orders
			plt.figure(figsize=(10, 6))
			
			# Iterate over each test
			for _TEST_NUM in test_range:
				# Load data from CSV files
				instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
				updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
				
				# Ensure arrays are of equal length for accurate comparison
				min_length = min(len(instron_data), len(updated_arduino_data))
				instron_force = instron_data["Force [N]"].iloc[:min_length]
				
				# Use this for meta-analysis of calibration
				# updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"].iloc[:min_length]
				
				# Use this for analysis of raw ADC with instron N
				updated_arduino_force = updated_arduino_data["ADC" if SIMPLIFY else f"ADC{sensor_num}"].iloc[:min_length]
				
				# Quantize input and output data
				instron_force = quantize_data(instron_force, bit_resolution)
				updated_arduino_force = quantize_data(updated_arduino_force, bit_resolution)
				
				# Apply polynomial fit to the data
				if segment_size is not None:  # Process data in segments
					# Initialize variables to store the combined fits and residuals
					combined_fits = np.zeros_like(instron_force)
					residuals_combined = np.zeros_like(instron_force)
					
					num_segments = (len(instron_force) + segment_size - 1) // segment_size  # Handles remainder segments
					mse_list, mae_list = [], []
					
					# Loop through each segment
					for i in range(num_segments):
						start = i * segment_size
						end = min(start + segment_size, len(instron_force))  # Ensure last segment doesn't go out of bounds
						
						instron_segment = instron_force[start:end]
						arduino_segment = updated_arduino_force[start:end]
						
						# Perform the polynomial fit on this segment
						if len(instron_segment) > order:  # Ensure enough points for the polynomial fit
							poly_coeffs = np.polyfit(instron_segment, arduino_segment, order)
							poly_fit = np.polyval(poly_coeffs, instron_segment)
						else:
							poly_fit = np.zeros_like(instron_segment)  # Handle short segments safely
						
						# Save the fit and calculate residuals
						combined_fits[start:end] = poly_fit
						residuals = arduino_segment - poly_fit
						residuals_combined[start:end] = residuals
						
						# Apply smoothing (if needed) to the residuals
						residuals_smoothed = apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
						
						# Calculate MSE and MAE for the segment
						mse = mean_squared_error(arduino_segment, poly_fit)
						mae = mean_absolute_error(arduino_segment, poly_fit)
						mse_list.append(mse)
						mae_list.append(mae)
						
						print(f"Test {_TEST_NUM}, Segment {i + 1}/{num_segments}, MSE={mse}, MAE={mae}")
					
					# Plot the combined polynomial fits and residuals
					plt.plot(instron_force, combined_fits, '-', label=f"Segmented Fit (Test {_TEST_NUM})", linewidth=2)
					
					if len(residuals_smoothed) == len(instron_force[:len(residuals_smoothed)]):
						plt.plot(instron_force[:len(residuals_smoothed)], residuals_smoothed, '--',
						         label=f"Smoothed Residuals (Test {_TEST_NUM})", linewidth=2)
				
				else:  # Full polynomial fit without segmentation
					# Fit the polynomial model
					lin_fit = calculate_line_of_best_fit(x=instron_force, y=updated_arduino_force, isPolyfit=True, order=order)
					residuals = updated_arduino_force - lin_fit
					
					# Calculate MSE and MAE for polynomial fit
					mse_poly = mean_squared_error(updated_arduino_force, lin_fit)
					mae_poly = mean_absolute_error(updated_arduino_force, lin_fit)
					
					print(f"Test {_TEST_NUM}, Polynomial Fit (Order {order}): MSE={mse_poly}, MAE={mae_poly}")
					
					# Apply smoothing using the specified method
					residuals_smoothed = apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
					
					# Plot the smoothed residuals
					plt.plot(instron_force[:len(residuals_smoothed)], residuals_smoothed, '-', label=f"Test {_TEST_NUM}", linewidth=2)
			
			# Finalize and display the plot
			plt.xlabel("Instron Force [N]")
			plt.ylabel("Arduino Force [ADC]")
			plt.legend(loc="lower left")
			plt.title(f"Polynomial Fit (Order {order}) Across Multiple Tests")
			plt.grid(True)
			plt.gca().invert_xaxis()
			
			if show_graphs:
				plt.show()
			
			if save_graphs:
				pdf.savefig()
			
			plt.close()
