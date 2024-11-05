import numpy as np
import tensorflow_model_optimization as tfmot
from keras.api.layers import BatchNormalization, Dense, Dropout
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from keras.api.regularizers import l2
from keras.api.utils import plot_model
from keras.src.layers import InputLayer


# Function to build a quantized neural network model
def build_quantized_neural_network(input_dim, layers=2, units=64, activation='relu', dropout_rate=0.5, l2_reg=0.01, learning_rate=0.001):
	"""
	Build a neural network with quantization-aware training at 8-bit resolution.
	"""
	# Define the basic sequential model
	model = Sequential()
	
	# Add input and first layer
	model.add(InputLayer(input_shape=(input_dim,)))
	model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))
	model.add(Dropout(dropout_rate))
	
	# Add more layers as needed
	for _ in range(layers - 1):
		model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))
		model.add(Dropout(dropout_rate))
	
	# Output layer
	model.add(Dense(1))  # Output layer for regression
	
	# Compile the model
	model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
	
	# Apply 8-bit quantization-aware training
	quant_aware_model = tfmot.quantization.keras.quantize_model(model)
	
	return quant_aware_model


def get_model_memory_usage(batch_size, model):
	import numpy as np
	from keras import backend as K
	
	shapes_mem_count = 0
	internal_model_mem_count = 0
	for l in model.layers:
		layer_type = l.__class__.__name__
		if layer_type == 'Model':
			internal_model_mem_count += get_model_memory_usage(batch_size, l)
		single_layer_mem = 1
		out_shape = l.output_shape
		if type(out_shape) is list:
			out_shape = out_shape[0]
		for s in out_shape:
			if s is None:
				continue
			single_layer_mem *= s
		shapes_mem_count += single_layer_mem
	
	trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
	non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
	
	number_size = 4.0
	if K.floatx() == 'float16':
		number_size = 2.0
	if K.floatx() == 'float64':
		number_size = 8.0
	
	total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
	gbytes = np.round(total_memory / (1024.0**3), 3) + internal_model_mem_count
	return gbytes


def get_bit_resolution(model):
	from keras import backend as K
	
	# Checking default precision used by Keras backend
	floatx_dtype = K.floatx()  # Returns 'float16', 'float32', or 'float64'
	bit_resolution = {'float16': 16, 'float32': 32, 'float64': 64}.get(floatx_dtype, 'Unknown')
	
	print(f"Model uses {bit_resolution}-bit precision for signals (weights, activations).")
	
	# If needed, we can also check each layer's dtype individually:
	for layer in model.layers:
		if hasattr(layer, 'dtype'):
			print(f"Layer {layer.name} uses {layer.dtype} precision")
	
	return bit_resolution


def get_quantized_bit_resolution(model):
	quantized_layers = []
	
	# Check each layer to determine if it's quantized
	for layer in model.layers:
		layer_type = layer.__class__.__name__
		
		# Layers commonly used in quantized models
		if 'Quant' in layer_type or hasattr(layer, 'quantize_config'):
			quantized_layers.append(layer)
			print(f"Layer {layer.name} is quantized.")
	
	if len(quantized_layers) == 0:
		print("No quantized layers found in the model.")
		return None
	
	# Assuming TensorFlow Model Optimization Toolkit or similar library is used
	for layer in quantized_layers:
		# Example of retrieving bit resolution - this part depends on the exact quantization method
		# Check for a custom attribute like 'quant_bits' (you may need to adjust this based on your framework)
		if hasattr(layer, 'quant_bits'):
			print(f"Layer {layer.name} uses {layer.quant_bits}-bit precision")
		else:
			print(f"Layer {layer.name} might use quantization, but bit precision is not directly accessible.")
	
	return quantized_layers


def determine_minimum_bit_resolution(data, precision=None, epsilon=1e-12):
	"""
	Determine the minimum bit resolution required to represent the input data
	as precisely as possible.

	Parameters:
	- data: NumPy array or list of input data to analyze.
	- precision: The required precision (smallest difference between values).
				 If None, it will be automatically calculated based on data.
	- epsilon: A small value to handle floating-point precision issues.

	Returns:
	- A dictionary containing:
		- minimum_bits: The minimum number of bits required.
		- dynamic_range: The range of the data.
		- min_val: The minimum value in the data.
		- max_val: The maximum value in the data.
		- precision: The precision used for calculation.
	"""
	
	# Convert to NumPy array if input is a list
	if isinstance(data, list):
		data = np.array(data)
	
	# Ensure data is a floating-point type for precision calculations
	data = data.astype(np.float64)
	
	# Calculate range
	min_val = np.min(data)
	max_val = np.max(data)
	dynamic_range = max_val - min_val
	
	# Handle case where dynamic range is zero (all values are identical)
	if dynamic_range == 0:
		print("All data points are identical. Minimum bit resolution is 1 bit.")
		return {
			"minimum_bits": 1,  # 1 bit is sufficient to represent a single unique value
			"dynamic_range": dynamic_range,
			"min_val": min_val,
			"max_val": max_val,
			"precision": 0  # No precision needed
		}
	
	# If precision is not provided, calculate it
	if precision is None:
		unique_values = np.unique(data)
		if len(unique_values) < 2:
			# Only one unique value exists
			print("Only one unique value found in data. Minimum bit resolution is 1 bit.")
			return {
				"minimum_bits": 1,
				"dynamic_range": dynamic_range,
				"min_val": min_val,
				"max_val": max_val,
				"precision": 0
			}
		# Calculate the smallest difference between sorted unique values
		diffs = np.diff(unique_values)
		precision = np.min(diffs)
		if not np.isfinite(precision) or precision <= 0:
			print("Calculated precision is non-finite or non-positive. Setting precision to epsilon.")
			precision = epsilon
	
	# Ensure precision is positive and finite
	if precision <= 0 or not np.isfinite(precision):
		print("Precision must be positive and finite. Setting precision to epsilon.")
		precision = epsilon
	
	# Calculate the number of bits required
	ratio = dynamic_range / precision
	if ratio <= 0:
		print("Dynamic range divided by precision is non-positive. Setting minimum bits to 1.")
		minimum_bits = 1
	else:
		log_ratio = np.log2(ratio)
		if not np.isfinite(log_ratio):
			print("Logarithm of ratio is non-finite. Setting minimum bits to 1.")
			minimum_bits = 1
		else:
			minimum_bits = int(np.ceil(log_ratio))
	
	return {
		"minimum_bits": minimum_bits,
		"dynamic_range": dynamic_range,
		"min_val": min_val,
		"max_val": max_val,
		"precision": precision
	}


def display_info(batch_size, model):
	model.summary(expand_nested=True, show_trainable=True)
	model_memory = get_model_memory_usage(batch_size, model)
	bit_resolution = get_bit_resolution(model)
	quantized_layers = get_quantized_bit_resolution(model)
	print(f"Total model memory usage: {model_memory} GB")
	print(f"Bit resolution: {bit_resolution} bits")
	print(f"Quantized layers: {quantized_layers}")
	
	# Print the weights for each layer
	print("Model Weights:")
	for layer in model.layers:
		weights = layer.get_weights()
		
		if len(weights) > 0:
			print(f"Layer: {layer.name}")
			# Different layers have different numbers of components
			if isinstance(layer, BatchNormalization):
				gamma, beta, moving_mean, moving_variance = weights
				print("Gamma (scale):", gamma)
				print("Beta (shift):", beta)
				print("Moving Mean:", moving_mean)
				print("Moving Variance:", moving_variance)
			elif len(weights) == 2:  # Dense layers have weights and biases
				weights, biases = weights
				print("Weights:", weights)
				print("Biases:", biases)
			print("-" * 50)


def display_layer_info(model):
	from matplotlib import pyplot as plt
	
	# Create a plot of the model architecture
	plot_model(model, to_file="model_visual.png", show_shapes=True, show_layer_names=True)
	
	# Display the generated plot
	img = plt.imread("model_visual.png")
	plt.figure(figsize=(10, 10))
	plt.imshow(img)
	plt.axis('off')
	plt.show()


def calculate_bit_resolution(data_title, data):
	# Determine minimum bit resolution for the data
	print(data_title + ":")
	result = determine_minimum_bit_resolution(data)
	print(f"Minimum Bit Resolution: {result['minimum_bits']} bits")
	print(f"Dynamic Range: {result['dynamic_range']}")
	print(f"Minimum Value: {result['min_val']}")
	print(f"Maximum Value: {result['max_val']}")
	print(f"Precision: {result['precision']}")


def plot_overlay(overlay_ax, inputs, targets, outputs, test_num, mapping):
	if mapping == 'N_vs_N':
		# Plot calibrated N values (Y) vs Instron N (X)
		x = targets.flatten()  # Instron N
		y_pred = outputs.flatten()  # Predicted N (Calibrated)
		xlabel = "Calibration Force [N]"
		ylabel = "Calibrated Force [N]"
	elif mapping == 'ADC_vs_N':
		# Plot ADC vs Instron N
		x = targets.flatten()  # Instron N
		y_pred = outputs.flatten()  # Predicted ADC
		xlabel = "Calibration Force [N]"
		ylabel = "ADC Value"
	else:
		raise ValueError("Invalid mapping type. Use 'ADC_vs_N' or 'N_vs_N'.")
	
	overlay_ax.plot(x, y_pred, label=f"Test {test_num} - Neural Fit", linewidth=2)
	overlay_ax.set_xlabel(xlabel)
	overlay_ax.set_ylabel(ylabel)
	overlay_ax.grid(True)


def plot_residuals(residuals_ax, instron_force, residuals, test_num, mapping):
	x = instron_force.flatten()  # Calibration Force (N) is the baseline
	
	# Invert the x-axis to make the direction go from larger to smaller
	residuals_ax.invert_xaxis()
	
	if mapping == 'N_vs_N':
		# First graph: Residuals in N vs Instron N
		residuals_ax.plot(x, residuals, label=f"Residuals [N] (Test {test_num})", linewidth=2)  # (Test {test_num})
		residuals_ax.set_xlabel("Calibration Force [N]")
		residuals_ax.set_ylabel("Residuals [N]")
	elif mapping == 'ADC_vs_N':
		# Second graph: Residuals in ADC vs Instron N
		residuals_ax.plot(x, residuals, label=f"Residuals [ADC] (Test {test_num})", linewidth=2)  # (Test {test_num})
		residuals_ax.set_xlabel("Calibration Force [N]")
		residuals_ax.set_ylabel("Residuals [ADC]")
	else:
		raise ValueError("Invalid mapping type. Use 'N_vs_N' or 'ADC_vs_N'.")
	residuals_ax.grid(True)

# # Analyze and plot function
# def analyze_and_graph_neural_fit_with_linear(
# 	test_range, sensor_num, units=32, layers=1, activation='relu', dropout_rate=0.2,
# 	l2_reg=0.001, learning_rate=0.0001, epochs=100, batch_size=32, save_graphs=True,
# 	show_graphs=True, bit_resolution=16, enable_hyperparameter_tuning=False,
# 	hyperparams_dict=None, save_bit=False, plot_4=False
# ):
# 	if hyperparams_dict is None:
# 		hyperparams_dict = {}
#
# 	plt.close('all')
# 	print(f"Neurons: {units}, bit resolution: {bit_resolution}")
#
# 	if save_bit:
# 		file_name = f"residuals_{bit_resolution}" if activation != "relu" else f"residuals_{bit_resolution}_relu"
# 	else:
# 		file_name = f"residuals_{units}_neurons" if activation != "relu" else f"residuals_{units}_neurons_relu"
#
# 	# Create new figures for the final calibrated output vs raw ADC plot
# 	residuals_2_fig, residuals_2_ax = plt.subplots(figsize=(10, 5))
#
# 	for test_num in test_range:
#
# 		# Load and prepare data (returns sensor data in N and Instron force)
# 		inputs, targets, _, _ = load_and_prepare_data_with_linear(sensor_num, test_num, bit_resolution)
#
# 		# Calculate linear fit coefficients
# 		m, b = calculate_linear_fit(targets, inputs)
#
# 		# Apply linear transformation to sensor data to map from ADC to N
# 		linear_predictions = m * inputs + b  # Linear predictions in N
#
# 		# Calculate residuals as the difference between actual Instron Force and linear predictions
# 		residuals_input = targets.flatten() - linear_predictions.flatten()
#
# 		# Train the model on the residuals
# 		if enable_hyperparameter_tuning:
# 			model, input_scaler, output_scaler, best_hyperparams = train_model_with_hyperparameter_tuning(inputs, residuals_input, bit_resolution, test_num, hyperparams_dict)
# 		else:
# 			# Train the model on the residuals, using the mapped sensor data as inputs and residuals as targets
# 			model, input_scaler, output_scaler = train_model(
# 				inputs, residuals_input, units, layers, activation, dropout_rate,
# 				l2_reg, learning_rate, epochs, batch_size, bit_resolution
# 			)
#
# 		# Evaluate the model to get residual corrections
# 		residual_corrections = evaluate_model(model, inputs, residuals_input, input_scaler, output_scaler)
#
# 		# Final calibrated output: linear fit + NN-predicted residuals
# 		final_calibrated_outputs = linear_predictions.flatten() + residual_corrections.flatten()
#
# 		# Calculate RMSE on the final calibrated outputs
# 		mse_nn = mean_squared_error(targets.flatten(), final_calibrated_outputs)
# 		rmse_nn = np.sqrt(mse_nn)
# 		print(f"Test {test_num}, Calibrated Fit (Linear + NN): RMSE={rmse_nn:.6f}")
#
# 		# Calculate RMSE for the base residuals (linear fit only)
# 		mse_base = mean_squared_error(targets.flatten(), linear_predictions.flatten())
# 		rmse_base = np.sqrt(mse_base)
# 		print(f"Test {test_num}, Base Residuals Quantized: RMSE={rmse_base:.6f}")
#
# 		if plot_4: # linear_predictions = mapped sensor
# 			calibrated_fig, calibrated_ax = plt.subplots(figsize=(10, 5))
# 			residuals_fig, residuals_ax = plt.subplots(figsize=(10, 5))
#
# 			# Plot final calibrated output vs raw sensor ADC vs calibration force
# 			calibrated_ax.scatter(targets.flatten(), linear_predictions.flatten(), label=f"Raw Sensor Data (Test {test_num - 8})", linewidth=2, color="black")
# 			calibrated_ax.plot(targets.flatten(), final_calibrated_outputs.flatten(), label=f"Calibrated Output (Test {test_num - 8})", linewidth=2, color="red")
#
# 			# Customize calibrated output plot visuals
# 			calibrated_ax.set_xlim([0, 1])
# 			calibrated_ax.set_ylim([0, 1])
# 			calibrated_ax.set_ylabel("Raw Sensor Output (N)", fontsize=SIZE_XXLARGE)
# 			setup_basic_plot(calibrated_ax, False)
#
# 			# Plot Base residuals (Quantized)
# 			residuals_ax.plot(targets.flatten(), -residuals_input, label=f"Test {test_num - 8}", linewidth=3, color="black")
#
# 			# Customize residuals graph visuals
# 			residuals_ax.set_xlim([0, 1])
# 			residuals_ax.set_ylabel(r"Raw $\epsilon$ (N)", fontsize=SIZE_XXXLARGE, labelpad=-5)
# 			setup_basic_plot(residuals_ax)
#
# 			# plt.close(residuals_fig)
# 			# plt.close(calibrated_fig)
#
# 		# Plot residuals after calibration
# 		residuals_2_ax.plot(targets.flatten(), targets.flatten() - final_calibrated_outputs.flatten(), label=f"Test {test_num - 8}", linewidth=3)
#
# 		# Customize residuals output plot visuals
# 		residuals_2_ax.set_xlim([0, 1])
# 		residuals_2_ax.set_ylabel(r"$\epsilon$ (N)", fontsize=SIZE_XXXLARGE, labelpad=-5)
# 		setup_basic_plot(residuals_2_ax)
#
# 		plt.tight_layout()
#
# 	# Save graphs
# 	if save_graphs:
# 		residuals_2_fig.savefig(f"/Users/jacobanderson/Documents/BYU Classes/Current BYU Classes/Research/Papers/{file_name}.pdf", dpi=300)
#
# 	# Show graphs
# 	if show_graphs:
# 		plt.show()
#
# 	plt.close(residuals_2_fig)

#
# def analyze_and_graph_calibrated_data_and_fits_single_pdf_combined_multiple_tests(
# 	test_range, sensor_num, smoothing_method=None, window_size=100, poly_order=None,
# 	save_graphs=True, show_graphs=True, bit_resolution=12, mapping='N_vs_N'
# ):
# 	"""
# 	Analyze and visualize residuals and polynomial fits of different orders for each sensor across multiple tests,
# 	combining all tests in one graph per polynomial order.
#
# 	Parameters:
# 	- test_range: A range or list of test numbers to include in the analysis.
# 	- sensor_num: The sensor number to analyze.
# 	- window_size: Window size for the smoothing operation. If None, no smoothing is applied.
# 	- poly_order: Polynomial order for the Savitzky-Golay filter.
# 	- smoothing_method: The smoothing method ('savgol', 'boxcar', or None).
# 	- save_graphs: Boolean flag to save the graphs as a PDF.
# 	- show_graphs: Boolean flag to display the graphs during the process.
# 	- bit_resolution: The bit resolution to quantize the data (default is 12 bits).
# 	- mapping: Defines what to map on the x and y axes. Options are 'N_vs_N' (Calibrated N vs Instron N)
# 			   or 'ADC_vs_N' (ADC vs Instron N).
# 	"""
# 	# Replace SENSOR_SET and STARTING_SENSOR with appropriate variables or parameters if needed
# 	SENSOR_SET = "YourSensorSet"  # Update with your actual sensor set identifier
# 	STARTING_SENSOR = sensor_num
#
# 	with PdfPages(f"/Users/jacobanderson/Downloads/Combined_Tests_Polynomial_Sensor_Analysis_Sensor_Set_{SENSOR_SET}_Sensor_{STARTING_SENSOR}.pdf") as pdf:
# 		for order in range(1, 5):
# 			plt.figure(figsize=(10, 6))
#
# 			# Iterate over each test
# 			for _TEST_NUM in test_range:
# 				# Load data from CSV files
# 				instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
# 				updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
#
# 				# Ensure arrays are of equal length for accurate comparison
# 				min_length = min(len(instron_data), len(updated_arduino_data))
# 				instron_force = instron_data["Force (N)"].iloc[:min_length]
#
# 				# Depending on mapping, choose data for updated_arduino_force and set labels
# 				if mapping == 'N_vs_N':
# 					# Use calibrated force data
# 					calibrated_force_column = "Force (N)" if SIMPLIFY else f"Force{sensor_num} (N)"
# 					if calibrated_force_column in updated_arduino_data.columns:
# 						updated_arduino_force = updated_arduino_data[calibrated_force_column].iloc[:min_length]
# 						ylabel = "Residual Force (N)"
# 					else:
# 						print(f"Calibrated force data not found for sensor {sensor_num} in test {_TEST_NUM}.")
# 						continue  # Skip this test if calibrated data is missing
# 				elif mapping == 'ADC_vs_N':
# 					# Use raw ADC data
# 					adc_column = "ADC" if SIMPLIFY else f"ADC{sensor_num}"
# 					if adc_column in updated_arduino_data.columns:
# 						updated_arduino_force = updated_arduino_data[adc_column].iloc[:min_length]
# 						ylabel = "Residual Force (ADC)"
# 					else:
# 						print(f"ADC data not found for sensor {sensor_num} in test {_TEST_NUM}.")
# 						continue  # Skip this test if ADC data is missing
# 				else:
# 					raise ValueError("Invalid mapping type. Use 'N_vs_N' or 'ADC_vs_N'.")
#
# 				# Quantize input and output data
# 				instron_force = quantize_data(instron_force, bit_resolution)
# 				updated_arduino_force = quantize_data(updated_arduino_force, bit_resolution)
#
# 				# Fit the polynomial model
# 				lin_fit = calculate_line_of_best_fit(x=instron_force, y=updated_arduino_force, isPolyfit=True, order=order)
# 				residuals = updated_arduino_force - lin_fit
#
# 				# Calculate MSE and MAE for polynomial fit
# 				# mse_poly = mean_squared_error(updated_arduino_force, lin_fit)
# 				# mae_poly = mean_absolute_error(updated_arduino_force, lin_fit)
# 				#
# 				# print(f"Test {_TEST_NUM}, Polynomial Fit (Order {order}): MSE={mse_poly:.6f}, MAE={mae_poly:.6f}")
#
# 				# Apply smoothing using the specified method
# 				residuals_smoothed = residuals  # apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
#
# 				# Plot the smoothed residuals
# 				plt.plot(instron_force[:len(residuals_smoothed)], residuals_smoothed, '-', label=f"Test {_TEST_NUM}", linewidth=2)
#
# 			plt.xlabel("Calibration Force (N)")
# 			plt.ylabel(ylabel)
# 			plt.legend(loc="lower left")
# 			plt.title(f"Residuals for Polynomial Fit (Order {order}) Across Multiple Tests")
# 			plt.grid(True)
# 			plt.gca().invert_xaxis()
#
# 			if show_graphs:
# 				plt.show()
#
# 			if save_graphs:
# 				pdf.savefig()
#
# 			plt.close()
