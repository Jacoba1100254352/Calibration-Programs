from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_squared_error

from Configuration_Variables import *
from Workflow_Programs.Supporting_and_Archived_Graphing_Functions.Supplemental_Sensor_Graph_Functions import *


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


def setup_basic_plot(plot, apply_sci=True):
	plot.tick_params(
		axis='both', which='major', labelsize=18, width=2.5, length=5, direction='in',
		labelcolor='black', pad=10, top=True, bottom=True, left=True, right=True
	)
	plt.setp(plot.get_xticklabels(), fontsize=SIZE_XXLARGE)
	plt.setp(plot.get_yticklabels(), fontsize=SIZE_XXLARGE)
	plot.grid(True, which='both', linestyle='-', linewidth=1.5)
	
	plot.legend(
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
	legend = plot.get_legend()
	legend.get_frame().set_linewidth(2.0)  # Increase the outline thickness
	
	if apply_sci:
		plot.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
		plot.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
		plot.yaxis.get_offset_text().set_size(SIZE_XLARGE)
	
	plt.tight_layout()


###    NOTE: THIS FUNCTION SHOULD BE CLEANED UP TO USE THE setup_basic_plot FUNCTION   ###
def analyze_and_graph_residuals_and_fits_individual_images(save_graphs=True, useArduinoADC=False, trim=False, plot_residuals=True):
	"""
	Analyze and export residuals and polynomial fits of different orders for each sensor into .mat files.
	"""
	for sensor_num in SENSORS_RANGE:
		# Sleep to avoid HTTPS request limit
		# time.sleep(5)
		
		# Load data from CSV files
		instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
		updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num))
		
		# Extract time, force, and ADC data
		updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]  # .to_numpy()
		
		# Get Aligned Arduino Data for ADC results to work regardless of SIMPLIFY's value
		aligned_arduino_data = pd.read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num))
		
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
		# trim_force_threshold = -0.005
		# truncate_force_threshold = -0.7
		# trim_index = instron_force.ge(trim_force_threshold).idxmin()
		# truncate_index = instron_force.ge(truncate_force_threshold).idxmin()
		
		# Truncate all datasets from this index onwards to ensure consistent lengths
		# instron_force = instron_force[trim_index:]
		# arduino_force = arduino_force[trim_index:]
		# instron_force = instron_force[:truncate_index]
		# arduino_force = arduino_force[:truncate_index]
		
		# Invert the Instron force
		instron_force = -instron_force
		
		# Calculate and plot the best-fit line
		lin_fit = calculate_line_of_best_fit(instron_force, arduino_force)  # instron_force
		
		# Plot the best-fit line over the scatter plot
		raw_fig, raw_ax = plt.subplots(figsize=(10, 6))
		plt.scatter(instron_force - min(instron_force), arduino_force - min(arduino_force), label="Data", color="black")
		plt.plot(instron_force - min(instron_force), lin_fit - min(lin_fit), label="Best-fit line", color="r", linewidth=2)
		
		# Set axis limits and grid
		if trim:
			raw_ax.set_xlim([0, 0.7])
		else:
			raw_ax.set_xlim([0, 1])
		# raw_ax.set_ylabel("Calibration Force (N)", fontsize=SIZE_LARGE, fontweight='bold', family='Helvetica Neue', labelpad=5)
		raw_ax.set_ylabel("Raw Sensor Output (N)", fontsize=SIZE_XXLARGE, labelpad=0)
		
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
		# ax = plt.gca()
		# ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
		# ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
		#
		# # Set larger font size for the scientific notation
		# ax.yaxis.get_offset_text().set_size(SIZE_XLARGE)  # Adjust the font size as needed
		
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
		if plot_residuals:
			for order in [1]:  # range(1, 5)
				# Sleep to avoid HTTPS request limit
				# time.sleep(2)
				
				# Fit the polynomial model # Both of these methods provide the same results :)
				# coefficients = np.polyfit(instron_force, arduino_force, order)
				# polynomial = np.poly1d(coefficients)
				# predicted_adc_force = polynomial(instron_force)
				
				lin_fit = calculate_line_of_best_fit(instron_force, arduino_force)  # instron_force
				residuals = arduino_force - lin_fit
				
				mse_nn = mean_squared_error(instron_force, arduino_force)  # = mean_squared_error(instron_force-lin_fit, residuals)
				rmse_nn = np.sqrt(mse_nn)
				print(f"RMSE={rmse_nn:.6f}")
				
				residuals_fig, residuals_ax = plt.subplots(figsize=(10, 6))
				plt.plot(instron_force - min(instron_force) / 2, residuals, '-', label=f"Residuals", color="black", linewidth=2)
				
				if order == 1:
					# For first-order, you might still want to plot the average line for comparison
					average_residual = np.mean(residuals)
					plt.axhline(y=average_residual, color='r', linestyle='-', label="Best-fit line", linewidth=2)  # Average Residual
				
				# plt.xlabel("Calibration Force (N)")
				# plt.ylabel("Sensor Error")
				
				# Set axis limits and grid
				if trim:
					residuals_ax.set_xlim([0, 0.7])
				else:
					residuals_ax.set_xlim([0, 1])
				residuals_ax.set_ylabel("Sensor Error (N)", fontsize=SIZE_XXLARGE, labelpad=-5)  # Sensor Error # $\epsilon$
				
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


def graph_sensor_data(save_graphs=True):
	"""
	Generate and save plots comparing force measurements from the Load Cell and calibrated sensors
	for each sensor. The plot includes the measured force over time and the difference between the two measurements.
	"""
	for sensor_num in SENSORS_RANGE:
		# Load data from CSV files
		instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num))
		updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num))
		
		# Extract time and force data
		instron_time, instron_force = instron_data["Time [s]"], instron_data["Force [N]"]
		updated_arduino_time = updated_arduino_data["Time [s]"]
		updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"]
		
		# Plotting force comparison
		plt.figure(figsize=(10, 6))
		plt.plot(updated_arduino_time, updated_arduino_force, label="Calibrated Sensor Force", color="red")
		plt.plot(instron_time, instron_force, label="Reference Force (Load Cell)", color="blue")
		difference = instron_force - updated_arduino_force
		plt.plot(instron_time, difference, label="Force Difference (Load Cell - Sensor)", color="green", linestyle="--")
		plt.xlabel("Time [s]")
		plt.ylabel("Force [N]")
		plt.legend()
		plt.title(f"Force Measurement Comparison")
		plt.grid(True)
		
		if save_graphs:
			plt.savefig(get_data_filepath(PLOTS_DIR, sensor_num), dpi=300)
		
		plt.show()
