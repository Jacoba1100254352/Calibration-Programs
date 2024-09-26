# from matplotlib.backends.backend_pdf import PdfPages
import random

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error

# from Configuration_Variables import *
# from Supplemental_Sensor_Graph_Functions import *
from Neural_Fit import *


seed_value = 42  # Or any other integer


def analyze_and_graph_neural_fit(
	test_range, sensor_num, units=64, layers=2, activation='tanh', dropout_rate=0.5, l2_reg=0.01,
	learning_rate=0.001, epochs=100, batch_size=32, window_size=None, poly_order=None,
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
			
			# Apply smoothing to residuals
			residuals_smoothed = apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
			
			# First Graph: Plot calibrated sensor N vs Instron N
			overlay_ax.plot(targets.flatten(), outputs.flatten(), label=f"Calibrated Sensor [N]", linestyle='--', linewidth=1)  # (Test {test_num})
			overlay_ax.plot(targets.flatten(), targets.flatten(), label=f"Instron [N]", linewidth=2)  # (Test {test_num})
			
			# Plot overlay (calibrated N values or ADC values vs Instron N)
			# plot_overlay(overlay_ax, inputs, targets, outputs, test_num, mapping)
			
			# Plot residuals
			plot_residuals(residuals_ax, instron_force, residuals_smoothed, test_num, mapping)
		
		# Finalize and save first graph (Calibrated N vs Instron N)
		overlay_ax.set_xlabel("Instron Force [N]")
		overlay_ax.set_ylabel("Calibrated Sensor Force [N]")
		overlay_ax.set_title(f"Calibrated Sensor [N] vs Baseline [N]")
		overlay_ax.legend(loc="upper left")
		overlay_ax.grid(True)
		overlay_ax.invert_xaxis()
		overlay_ax.invert_yaxis()
		if save_graphs:
			pdf.savefig(overlay_fig)
		
		residuals_ax.set_title(f"Residuals with {bit_resolution}-bit model")
		residuals_ax.legend(loc="upper left")
		if save_graphs:
			pdf.savefig(residuals_fig)
		
		if show_graphs:
			plt.show()
		plt.close(overlay_fig)
		plt.close(residuals_fig)


def analyze_and_dual_graph_neural_fit(
	test_range, sensor_num, units=64, layers=2, activation='relu', dropout_rate=0.5, l2_reg=0.01,
	learning_rate=0.001, epochs=100, batch_size=32, window_size=None, poly_order=None,
	smoothing_method="boxcar", save_graphs=True, show_graphs=True, bit_resolution=12,
	enable_hyperparameter_tuning=False, hyperparams_dict=None
):
	"""
	Analyze and visualize neural network fits for each test and compare them across multiple tests.
	Generates:
	1. Graph comparing calibrated sensor N vs Instron N.
	2. Dual subplots:
	   - Residuals in N vs Instron N (N_vs_N).
	   - Residuals in ADC vs Instron N (ADC_vs_N).
	"""
	plt.close('all')
	
	# Initialize the PDF to save the graphs
	with PdfPages(f"/Users/jacobanderson/Downloads/Neural_Network_Fit_Sensor_Set_{sensor_num}.pdf") as pdf:
		# Set up figures and axes
		overlay_fig, overlay_ax = plt.subplots(figsize=(10, 6))  # First graph: Calibrated N vs Instron N
		residuals_fig, (residuals_n_ax, residuals_adc_ax) = plt.subplots(2, 1, figsize=(10, 12))  # Dual subplots
		
		for test_num in test_range:
			# Load and prepare data
			inputs_adc_vs_n, targets_adc_vs_n, instron_force, sensor_adc = load_and_prepare_data(
				sensor_num, test_num, bit_resolution, mapping='ADC_vs_N'
			)
			
			inputs_n_vs_n, targets_n_vs_n, _, _ = load_and_prepare_data(
				sensor_num, test_num, bit_resolution, mapping='N_vs_N'
			)
			
			if enable_hyperparameter_tuning:
				# Train model with hyperparameter tuning for both mappings
				model_adc_vs_n, input_scaler_adc_vs_n, output_scaler_adc_vs_n, _ = train_model_with_hyperparameter_tuning(
					inputs_adc_vs_n, targets_adc_vs_n, bit_resolution, test_num, hyperparams_dict
				)
				model_n_vs_n, input_scaler_n_vs_n, output_scaler_n_vs_n, _ = train_model_with_hyperparameter_tuning(
					inputs_n_vs_n, targets_n_vs_n, bit_resolution, test_num, hyperparams_dict
				)
			else:
				# Train model without hyperparameter tuning
				model_adc_vs_n, input_scaler_adc_vs_n, output_scaler_adc_vs_n = train_model(
					inputs_adc_vs_n, targets_adc_vs_n, units, layers, activation, dropout_rate, l2_reg,
					learning_rate, epochs, batch_size, bit_resolution
				)
				model_n_vs_n, input_scaler_n_vs_n, output_scaler_n_vs_n = train_model(
					inputs_n_vs_n, targets_n_vs_n, units, layers, activation, dropout_rate, l2_reg,
					learning_rate, epochs, batch_size, bit_resolution
				)
			
			# Ensure reproducibility
			torch.manual_seed(seed_value)
			np.random.seed(seed_value)
			random.seed(seed_value)
			if torch.cuda.is_available():
				torch.cuda.manual_seed(seed_value)
				torch.cuda.manual_seed_all(seed_value)
			
			# Evaluate models for both mappings
			outputs_adc_vs_n, residuals_adc = evaluate_model(
				model_adc_vs_n, inputs_adc_vs_n, instron_force, sensor_adc, input_scaler_adc_vs_n, output_scaler_adc_vs_n, mapping='ADC_vs_N'
			)
			
			outputs_n_vs_n, residuals_n = evaluate_model(
				model_n_vs_n, inputs_n_vs_n, instron_force, sensor_adc, input_scaler_n_vs_n, output_scaler_n_vs_n, mapping='N_vs_N'
			)
			
			# Apply smoothing to residuals
			residuals_n_smoothed = apply_smoothing(residuals_n, method=smoothing_method, window_size=window_size, poly_order=poly_order)
			residuals_adc_smoothed = apply_smoothing(residuals_adc, method=smoothing_method, window_size=window_size, poly_order=poly_order)
			
			# First Graph: Plot calibrated sensor N vs Instron N
			overlay_ax.plot(instron_force.flatten(), outputs_n_vs_n.flatten(), label=f"Calibrated Sensor N (Test {test_num})", linestyle='--', linewidth=1)
			overlay_ax.plot(instron_force.flatten(), instron_force.flatten(), label=f"Instron N (Test {test_num})", linewidth=2)
			
			# Second Graph (Subplot 1): Residuals in N (N_vs_N)
			residuals_n_ax.plot(instron_force.flatten(), residuals_n_smoothed, label=f"Residuals in N (Test {test_num})", linewidth=2)
			
			# Second Graph (Subplot 2): Residuals in ADC (ADC_vs_N)
			residuals_adc_ax.plot(instron_force.flatten(), residuals_adc_smoothed, label=f"Residuals in ADC (Test {test_num})", linewidth=2)
		
		# Finalize and save first graph (Calibrated N vs Instron N)
		overlay_ax.set_xlabel("Instron Force [N]")
		overlay_ax.set_ylabel("Force [N]")
		overlay_ax.set_title(f"Calibrated Sensor N vs Instron N for Tests {test_range}")
		overlay_ax.legend(loc="upper left")
		overlay_ax.grid(True)
		overlay_ax.invert_xaxis()
		overlay_ax.invert_yaxis()
		if save_graphs:
			pdf.savefig(overlay_fig)
		
		# Finalize and save residuals subplot (N_vs_N)
		residuals_n_ax.set_xlabel("Instron Force [N]")
		residuals_n_ax.set_ylabel("Residuals in N")
		residuals_n_ax.set_title(f"Residuals in N for Tests {test_range}")
		residuals_n_ax.legend(loc="upper left")
		residuals_n_ax.grid(True)
		residuals_n_ax.invert_xaxis()
		
		# Finalize and save residuals subplot (ADC_vs_N)
		residuals_adc_ax.set_xlabel("Instron Force [N]")
		residuals_adc_ax.set_ylabel("Residuals in ADC")
		residuals_adc_ax.set_title(f"Residuals in ADC for Tests {test_range}")
		residuals_adc_ax.legend(loc="upper left")
		residuals_adc_ax.grid(True)
		residuals_adc_ax.invert_xaxis()
		
		# Save both subplots
		if save_graphs:
			pdf.savefig(residuals_fig)
		
		# Show graphs if requested
		if show_graphs:
			plt.show()
		
		# Close figures
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
