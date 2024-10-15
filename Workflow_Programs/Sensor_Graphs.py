# from matplotlib.backends.backend_pdf import PdfPages
import time

from sklearn.metrics import mean_absolute_error

# from Configuration_Variables import *
# from Supplemental_Sensor_Graph_Functions import *
from Neural_Fit import *


seed_value = 42

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages


# Set general plot appearance
SIZE_SMALL = 10
SIZE_DEFAULT = 14
SIZE_LARGE = 20
SIZE_XLARGE = 26
SIZE_XXLARGE = 32
SIZE_XXXLARGE = 40

plt.rc("font", family='Helvetica Neue', size=SIZE_DEFAULT, weight="bold")  # Default text sizes
plt.rc("axes", labelsize=SIZE_LARGE)  # X and Y labels fontsize
plt.rc("axes", linewidth=2.5)  # Line width for plot borders


# Analyze and plot function
def analyze_and_graph_neural_fit(
	test_range, sensor_num, units=64, layers=2, activation='tanh', dropout_rate=0.5,
	l2_reg=0.01, learning_rate=0.001, epochs=100, batch_size=32, save_graphs=True,
	show_graphs=True, bit_resolution=12, enable_hyperparameter_tuning=False, mapping='N_vs_N',
	hyperparams_dict=None, save_bit=False
):
	plt.close('all')
	print(f"Neurons: {units}, bit resolution: {bit_resolution}")
	
	if save_bit:
		file_name = f"residuals_{bit_resolution}" if activation != "relu" else f"residuals_{bit_resolution}_relu"
	else:
		file_name = f"residuals_{units}_neurons" if activation != "relu" else f"residuals_{units}_neurons_relu"
	
	residuals_fig, residuals_ax = plt.subplots(figsize=(10, 5))
	
	for test_num in test_range:
		
		# Load and prepare data (placeholder function call)
		inputs, targets, instron_force, sensor_adc = load_and_prepare_data(sensor_num, test_num, bit_resolution, mapping)
		
		# Train model and evaluate (placeholder function call)
		model, input_scaler, output_scaler = train_model(inputs, targets, units, layers, activation, dropout_rate, l2_reg, learning_rate, epochs, batch_size, bit_resolution)
		outputs, residuals = evaluate_model(model, inputs, instron_force, sensor_adc, input_scaler, output_scaler, mapping)
		
		# Calculate RMSE
		mse_nn = mean_squared_error(targets.flatten(), outputs.flatten())
		rmse_nn = np.sqrt(mse_nn)
		print(f"Test {test_num}, Neural Network Fit: RMSE={rmse_nn:.6f}")
		
		# Plot residuals
		residuals_ax.plot(instron_force.flatten(), residuals, label=f"Test {test_num-8}", linewidth=3)
		
		# Set axis limits and grid
		residuals_ax.set_xlim([0, 1])
		residuals_ax.set_ylim([-0.04, 0.06])
		residuals_ax.set_ylabel("$\epsilon$ (N)", fontsize=SIZE_XXXLARGE, labelpad=-5)
		# residuals_ax.set_xlabel("Calibration Force (N)", fontsize=SIZE_LARGE, fontweight='bold', family='Helvetica Neue', labelpad=5)  # Bold label
		
		# Bold and increase size of the tick labels
		residuals_ax.tick_params(
			axis='both', which='major', labelsize=18, width=2.5, length=5, direction='in',
			labelcolor='black', pad=10, top=True, bottom=True, left=True, right=True
		)  # Major ticks on all sides
		residuals_ax.tick_params(
			axis='both', which='minor', labelsize=14, width=1.5, length=2.5, direction='in',
			labelcolor='black', top=True, bottom=True, left=True, right=True
		)  # Minor ticks on all sides
		
		# Apply bold and Helvetica to tick labels using setp() # (Numbers used)
		plt.setp(residuals_ax.get_xticklabels(), fontsize=SIZE_XXLARGE)  # X ticks
		plt.setp(residuals_ax.get_yticklabels(), fontsize=SIZE_XXLARGE)  # Y ticks
		
		# SMALL TICKS
		# Add minor ticks
		# residuals_ax.xaxis.set_minor_locator(AutoMinorLocator())
		# residuals_ax.yaxis.set_minor_locator(AutoMinorLocator())
		
		# GRID LINES
		# Set grid with minor ticks and tick-like lines
		residuals_ax.grid(True, which='both', linestyle='-', linewidth=1.5)  # Minor and major grid lines
		
		# Add minor tick marks inside the graph
		# residuals_ax.tick_params(which='minor', length=5, width=1.5, direction='in')  # Shorter ticks for minor grid lines
		# residuals_ax.tick_params(which='major', length=10, width=2.5, direction='in')  # Longer ticks for major grid lines
		
		# Formatter for scientific notation
		ax = plt.gca()
		ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
		ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
		
		# Set larger font size for the scientific notation
		ax.yaxis.get_offset_text().set_size(SIZE_XLARGE)  # Adjust the font size as needed
		
		plt.tight_layout()
		
		# Add legend
		# residuals_ax.legend(loc="upper right", fontsize=SIZE_LARGE)
		residuals_ax.legend(
			loc="upper right",
			# fontsize=SIZE_DEFAULT,
			prop={'family': 'Helvetica Neue', 'size': SIZE_DEFAULT},  # Set font to Helvetica
			frameon=True,  # Enable the frame (box around the legend)
			edgecolor='black',  # Set the outline color
			framealpha=1,  # Set the transparency of the frame (1 = fully opaque)
			fancybox=False,  # Disable rounded corners
			shadow=False,  # No shadow
			facecolor='white',  # Background color of the legend box
			borderpad=0.5  # Padding inside the legend box
		)
		
		# Set the thickness of the legend box outline (bold)
		legend = residuals_ax.get_legend()
		legend.get_frame().set_linewidth(2.0)  # Increase the outline thickness
	
	# Save graphs
	if save_graphs:
		plt.savefig(f"/Users/jacobanderson/Documents/BYU Classes/Current BYU Classes/Research/Papers/{file_name}.pdf", dpi=300)
	
	# Show graphs
	if show_graphs:
		plt.show()
	
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
				instron_force = instron_data["Force (N)"].iloc[:min_length]
				
				# Depending on mapping, choose data for updated_arduino_force and set labels
				if mapping == 'N_vs_N':
					# Use calibrated force data
					calibrated_force_column = "Force (N)" if SIMPLIFY else f"Force{sensor_num} (N)"
					if calibrated_force_column in updated_arduino_data.columns:
						updated_arduino_force = updated_arduino_data[calibrated_force_column].iloc[:min_length]
						ylabel = "Residual Force (N)"
					else:
						print(f"Calibrated force data not found for sensor {sensor_num} in test {_TEST_NUM}.")
						continue  # Skip this test if calibrated data is missing
				elif mapping == 'ADC_vs_N':
					# Use raw ADC data
					adc_column = "ADC" if SIMPLIFY else f"ADC{sensor_num}"
					if adc_column in updated_arduino_data.columns:
						updated_arduino_force = updated_arduino_data[adc_column].iloc[:min_length]
						ylabel = "Residual Force (ADC)"
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
				residuals_smoothed = residuals  # apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
				
				# Plot the smoothed residuals
				plt.plot(instron_force[:len(residuals_smoothed)], residuals_smoothed, '-', label=f"Test {_TEST_NUM}", linewidth=2)
			
			plt.xlabel("Calibration Force (N)")
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
		# instron_time = instron_data["Time [s]"]  # .to_numpy()
		instron_force = instron_data["Force [N]"]  # .to_numpy()
		# updated_arduino_time = updated_arduino_data["Time [s]"]  # .to_numpy()
		updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]  # .to_numpy()
		
		# Get Aligned Arduino Data for ADC results to work regardless of SIMPLIFY's value
		aligned_arduino_data = pd.read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num))
		
		# Calculate force difference for export if needed
		# difference = instron_force - updated_arduino_force
		
		# Ensure arrays are of equal length for accurate comparison
		min_length = min(len(instron_data), len(updated_arduino_data))
		instron_force = instron_data["Force [N]"].iloc[:min_length]
		if useArduinoADC:
			arduino_force_type = "ADC" if SIMPLIFY else f"ADC{sensor_num}"
			arduino_force = aligned_arduino_data["ADC" if SIMPLIFY else f"ADC{sensor_num}"].iloc[:min_length]
		else:
			arduino_force_type = "Force (N)" if SIMPLIFY else f"Force{sensor_num} (N)"
			arduino_force = -updated_arduino_force.iloc[:min_length]
		
		# Find the index where the Instron force first reaches the threshold
		trim_force_threshold = -0.005
		truncate_force_threshold = -0.7
		trim_index = instron_force.ge(trim_force_threshold).idxmin()
		truncate_index = instron_force.ge(truncate_force_threshold).idxmin()
		
		# Truncate all datasets from this index onwards to ensure consistent lengths
		instron_force = instron_force[trim_index:]
		arduino_force = arduino_force[trim_index:]
		instron_force = instron_force[:truncate_index]
		arduino_force = arduino_force[:truncate_index]
		
		# Invert the Instron force
		instron_force = -instron_force
		
		# Second plot: Relationship between Instron force and Arduino ADC values
		# plt.figure(figsize=(10, 6))
		# plt.scatter(instron_force, arduino_force, label=f"Calibration Force vs. Pressure Sensor Output", color="purple")
		# plt.xlabel("Calibration Force [N]")
		# plt.ylabel(f"Pressure Sensor Output [counts]")
		# plt.legend()
		# plt.title(
		# 	f"Relationship Between Calibration Force and Pressure Sensor Count")
		# plt.grid(True)
		#
		# if save_graphs:
		# 	plt.savefig(f"/Users/jacobanderson/Downloads/Test {TEST_NUM} Sensor {sensor_num} adc against N.png", dpi=300)
		# plt.show()
		
		# Calculate and plot the best-fit line
		lin_fit = calculate_line_of_best_fit(instron_force, arduino_force)  # instron_force
		
		# Plot the best-fit line over the scatter plot
		raw_fig, raw_ax = plt.subplots(figsize=(10, 6))
		plt.scatter(instron_force, arduino_force - min(arduino_force), label="Data", color="black")
		plt.plot(instron_force, lin_fit - min(lin_fit), label="Best-fit line", color="r", linewidth=2)
		
		# Set axis limits and grid
		raw_ax.set_xlim([0, 0.7])
		# raw_ax.set_ylabel("Calibration Force (N)", fontsize=SIZE_LARGE, fontweight='bold', family='Helvetica Neue', labelpad=5)
		raw_ax.set_ylabel("Raw Pressure Sensor Output", fontsize=SIZE_XXLARGE, labelpad=0)
		
		# Bold and increase size of the tick labels
		raw_ax.tick_params(
			axis='both', which='major', labelsize=18, width=2.5, length=5, direction='in',
			labelcolor='black', pad=10, top=True, bottom=True, left=True, right=True
		)  # Major ticks on all sides
		raw_ax.tick_params(
			axis='both', which='minor', labelsize=14, width=1.5, length=2.5, direction='in',
			labelcolor='black', top=True, bottom=True, left=True, right=True
		)  # Minor ticks on all sides
		
		# Apply bold and Helvetica to tick labels using setp() # (Numbers used)
		plt.setp(raw_ax.get_xticklabels(), fontsize=SIZE_XXLARGE)  # X ticks
		plt.setp(raw_ax.get_yticklabels(), fontsize=SIZE_XXLARGE)  # Y ticks
		
		# SMALL TICKS
		# Add minor ticks
		# raw_ax.xaxis.set_minor_locator(AutoMinorLocator())
		# raw_ax.yaxis.set_minor_locator(AutoMinorLocator())
		
		# GRID LINES
		# Set grid with minor ticks and tick-like lines
		raw_ax.grid(True, which='both', linestyle='-', linewidth=1.5)  # Minor and major grid lines
		
		# Add minor tick marks inside the graph
		# raw_ax.tick_params(which='minor', length=5, width=1.5, direction='in')  # Shorter ticks for minor grid lines
		# raw_ax.tick_params(which='major', length=10, width=2.5, direction='in')  # Longer ticks for major grid lines
		
		# Formatter for scientific notation
		ax = plt.gca()
		ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
		ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
		
		# Set larger font size for the scientific notation
		ax.yaxis.get_offset_text().set_size(SIZE_XLARGE)  # Adjust the font size as needed
		
		plt.tight_layout()
		
		# plt.legend()
		raw_ax.legend(
			# loc="upper right",
			# fontsize=SIZE_DEFAULT,
			prop={'family': 'Helvetica Neue', 'size': SIZE_LARGE},  # Set font to Helvetica
			frameon=True,  # Enable the frame (box around the legend)
			edgecolor='black',  # Set the outline color
			framealpha=1,  # Set the transparency of the frame (1 = fully opaque)
			fancybox=False,  # Disable rounded corners
			shadow=False,  # No shadow
			facecolor='white',  # Background color of the legend box
			borderpad=0.5  # Padding inside the legend box
		)
		
		# Set the thickness of the legend box outline (bold)
		legend = raw_ax.get_legend()
		legend.get_frame().set_linewidth(2.0)  # Increase the outline thickness
		
		if save_graphs:
			file_name = "best_fit"
			plt.savefig(f"/Users/jacobanderson/Documents/BYU Classes/Current BYU Classes/Research/Papers/{file_name}.pdf", dpi=300)
		plt.show()
		
		# Calculate and plot residuals
		# residuals = arduino_force - lin_fit
		# print("Length of residuals:", len(residuals))
		# plt.figure(figsize=(10, 6))
		# plt.scatter(instron_force, residuals, label="Residuals", color="green")
		
		# plt.xlabel("Calibration Force [N]")
		# plt.ylabel("Residuals")
		# plt.legend()
		# plt.title(
		# 	f"Smoothed Residuals of {arduino_force_type} Values for {SENSOR_SET_DIR}, Sensor {sensor_num}, Test {TEST_NUM}")
		# plt.grid(True)
		#
		# if save_graphs:
		# 	plt.savefig(f"/Users/jacobanderson/Downloads/Test {TEST_NUM} Sensor {sensor_num} averaged noise.png", dpi=300)
		# plt.show()
		
		# Fit and plot polynomial models of 1st through 4th order
		for order in [1]:  # range(1, 5)
			# Sleep to avoid HTTPS request limit
			time.sleep(2)
			
			# Fit the polynomial model
			coefficients = np.polyfit(instron_force, arduino_force, order)
			polynomial = np.poly1d(coefficients)
			predicted_adc_force = polynomial(instron_force)
			residuals = arduino_force - predicted_adc_force
			
			residuals_fig, residuals_ax = plt.subplots(figsize=(10, 6))
			plt.plot(instron_force - min(instron_force) / 2, residuals, '-', label=f"Residuals", color="black", linewidth=2)
			
			if order == 1:
				# For first-order, you might still want to plot the average line for comparison
				average_residual = np.mean(residuals)
				plt.axhline(y=average_residual, color='r', linestyle='-', label="Best-fit line", linewidth=2)  # Average Residual
			
			# plt.xlabel("Calibration Force (N)")
			# plt.ylabel("Sensor Error")
			
			# Set axis limits and grid
			residuals_ax.set_xlim([0, 0.7])
			# residuals_ax.set_ylabel("Calibration Force (N)", fontsize=SIZE_LARGE, fontweight='bold', family='Helvetica Neue', labelpad=5)
			residuals_ax.set_ylabel("Sensor Error", fontsize=SIZE_XXLARGE, labelpad=-5)  # Sensor Error # $\epsilon$
			
			# Bold and increase size of the tick labels
			residuals_ax.tick_params(
				axis='both', which='major', labelsize=18, width=2.5, length=5, direction='in',
				labelcolor='black', pad=10, top=True, bottom=True, left=True, right=True
			)  # Major ticks on all sides
			residuals_ax.tick_params(
				axis='both', which='minor', labelsize=14, width=1.5, length=2.5, direction='in',
				labelcolor='black', top=True, bottom=True, left=True, right=True
			)  # Minor ticks on all sides
			
			# Apply bold and Helvetica to tick labels using setp()
			plt.setp(residuals_ax.get_xticklabels(), fontsize=SIZE_XXLARGE)  # X ticks
			plt.setp(residuals_ax.get_yticklabels(), fontsize=SIZE_XXLARGE)  # Y ticks
			
			# SMALL TICKS
			# Add minor ticks
			# residuals_ax.xaxis.set_minor_locator(AutoMinorLocator())
			# residuals_ax.yaxis.set_minor_locator(AutoMinorLocator())
			
			# GRID LINES
			# Set grid with minor ticks and tick-like lines
			residuals_ax.grid(True, which='both', linestyle='-', linewidth=1.5)  # Minor and major grid lines
			
			# Add minor tick marks inside the graph
			# residuals_ax.tick_params(which='minor', length=5, width=1.5, direction='in')  # Shorter ticks for minor grid lines
			# residuals_ax.tick_params(which='major', length=10, width=2.5, direction='in')  # Longer ticks for major grid lines
			
			# Formatter for scientific notation
			ax = plt.gca()
			ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
			ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
			
			# Set larger font size for the scientific notation
			ax.yaxis.get_offset_text().set_size(SIZE_XLARGE)  # Adjust the font size as needed
			
			# plt.legend()
			residuals_ax.legend(
				# loc="upper right",
				# fontsize=SIZE_DEFAULT,
				prop={'family': 'Helvetica Neue', 'size': SIZE_LARGE},  # Set font to Helvetica
				frameon=True,  # Enable the frame (box around the legend)
				edgecolor='black',  # Set the outline color
				framealpha=1,  # Set the transparency of the frame (1 = fully opaque)
				fancybox=False,  # Disable rounded corners
				shadow=False,  # No shadow
				facecolor='white',  # Background color of the legend box
				borderpad=0.5  # Padding inside the legend box
			)
			
			# Set the thickness of the legend box outline (bold)
			legend = residuals_ax.get_legend()
			legend.get_frame().set_linewidth(2.0)  # Increase the outline thickness
			
			# plt.title(f"Residuals: Error")
			plt.grid(True)
			
			plt.tight_layout()
			
			if save_graphs:
				file_name = "first_order_removed"
				plt.savefig(f"/Users/jacobanderson/Documents/BYU Classes/Current BYU Classes/Research/Papers/{file_name}.pdf", dpi=300)
			plt.show()
