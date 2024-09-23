import brevitas.nn as qnn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import torch.nn as nn
from brevitas.core.scaling import ScalingImplType
from brevitas.quant import Int8WeightPerTensorFixedPoint
from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.signal import medfilt, savgol_filter
from scipy.stats import linregress
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


class QuantizedNN(nn.Module):
	def __init__(
		self, input_dim, units=64, layers=2, activation='relu', dropout_rate=0.5,
		weight_bit_width=8, act_bit_width=8
	):
		super(QuantizedNN, self).__init__()
		
		# Define activations
		activation_functions = {
			'relu': qnn.QuantReLU,
			'tanh': qnn.QuantTanh,
			'sigmoid': qnn.QuantSigmoid
		}
		
		quant_activation_class = activation_functions.get(activation.lower(), qnn.QuantReLU)
		
		# Input layer with automatic quantization scaling
		self.layers = nn.ModuleList([
			qnn.QuantLinear(
				input_dim, units,
				bias=True,
				weight_bit_width=weight_bit_width,
				weight_quant=Int8WeightPerTensorFixedPoint,
				scaling_impl_type=ScalingImplType.STATS
			)
		])
		self.activations = nn.ModuleList([quant_activation_class(bit_width=act_bit_width)])
		self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate)])
		
		# Hidden layers
		for _ in range(1, layers):
			self.layers.append(
				qnn.QuantLinear(
					units, units,
					bias=True,
					weight_bit_width=weight_bit_width,
					weight_quant=Int8WeightPerTensorFixedPoint,
					scaling_impl_type=ScalingImplType.STATS
				)
			)
			self.activations.append(quant_activation_class(bit_width=act_bit_width))
			self.dropouts.append(nn.Dropout(dropout_rate))
		
		# Output layer
		self.output_layer = qnn.QuantLinear(
			units, 1,
			bias=True,
			weight_bit_width=weight_bit_width,
			weight_quant=Int8WeightPerTensorFixedPoint,
			scaling_impl_type=ScalingImplType.STATS
		)
	
	def forward(self, x):
		for layer, activation, dropout in zip(self.layers, self.activations, self.dropouts):
			x = layer(x)
			x = activation(x)
			x = dropout(x)
		x = self.output_layer(x)
		return x


def avg(lst):
	return sum(lst) / len(lst)


def difference_polarity(lst1, lst2):
	return (avg(lst1) - avg(lst2)) / abs(avg(lst1) - avg(lst2))


def calculate_line_of_best_fit(x, y, isPolyfit=False, order=1):
	if not isPolyfit:
		slope_avg, intercept_avg, _, _, _ = linregress(x, y)
		line_of_best_fit = slope_avg * x + intercept_avg
	else:
		coefficients = np.polyfit(x, y, order)
		polynomial = np.poly1d(coefficients)
		line_of_best_fit = polynomial(x)
	return line_of_best_fit


# Function to apply smoothing to residuals
def apply_smoothing(residuals, method, window_size, poly_order):
	"""
	Apply smoothing to the residuals using the specified method.

	Parameters:
	- residuals: The residual data to be smoothed.
	- method: The smoothing method ('savgol', 'boxcar', 'median', or None).
	- window_size: The window size for the smoothing operation.
	- poly_order: The polynomial order for Savitzky-Golay filter (only used if method is 'savgol').

	Returns:
	- smoothed_residuals: The smoothed residuals.
	"""
	if isinstance(residuals, pd.Series):
		residuals = residuals.values.flatten()  # Convert Pandas Series to NumPy array and flatten
	else:
		residuals = residuals.flatten()  # If it's already a NumPy array, just flatten it
	
	if method == 'savgol':
		if window_size is None:
			raise ValueError("Window size must be specified for Savitzky-Golay smoothing.")
		smoothed_residuals = savgol_filter(residuals, window_length=window_size, polyorder=poly_order)
	elif method == 'boxcar':
		if window_size is None:
			raise ValueError("Window size must be specified for boxcar smoothing.")
		smoothed_residuals = np.convolve(residuals, np.ones(window_size) / window_size, mode='valid')
		smoothed_residuals = np.pad(smoothed_residuals, (window_size // 2, window_size // 2), mode='edge')
		if len(smoothed_residuals) > len(residuals):
			smoothed_residuals = smoothed_residuals[:len(residuals)]
		elif len(smoothed_residuals) < len(residuals):
			smoothed_residuals = np.pad(smoothed_residuals, (0, len(residuals) - len(smoothed_residuals)), 'edge')
	elif method == 'median':
		if window_size is None:
			raise ValueError("Window size must be specified for median filtering.")
		if window_size % 2 == 0:  # Ensure window_size is odd
			window_size += 1
		smoothed_residuals = medfilt(residuals, kernel_size=window_size)
	else:
		smoothed_residuals = residuals
	
	return smoothed_residuals


# Helper function for quantizing data (if using custom bit resolutions for inputs/outputs)
def quantize_data(data, bit_resolution):
	"""Quantize the data to the given bit resolution."""
	max_val = np.max(np.abs(data))
	scale = (2**bit_resolution) - 1
	quantized_data = np.round(data / max_val * scale) * max_val / scale
	return quantized_data


# Function to build a quantized neural network model
def build_quantized_neural_network(input_dim, layers=2, units=64, activation='relu', dropout_rate=0.5, l2_reg=0.01, learning_rate=0.001):
	"""
	Build a neural network with quantization-aware training at 8-bit resolution.
	"""
	# Define the basic sequential model
	model = tf.keras.Sequential()
	
	# Add input and first layer
	model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
	model.add(tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
	model.add(tf.keras.layers.Dropout(dropout_rate))
	
	# Add more layers as needed
	for _ in range(layers - 1):
		model.add(tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
		model.add(tf.keras.layers.Dropout(dropout_rate))
	
	# Output layer
	model.add(tf.keras.layers.Dense(1))  # Output layer for regression
	
	# Compile the model
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
	
	# Apply 8-bit quantization-aware training
	quant_aware_model = tfmot.quantization.keras.quantize_model(model)
	
	return quant_aware_model


# Function to build a neural network model
def build_neural_network(input_dim, layers=2, units=64, activation='relu', dropout_rate=0.5, l2_reg=0.01, learning_rate=0.001):
	"""
	Build a customizable neural network model with specified parameters.

	Parameters:
	- input_dim: Dimension of the input data.
	- layers: Number of hidden layers in the neural network.
	- units: Number of units in each hidden layer.
	- activation: Activation function for hidden layers.
	- dropout_rate: Dropout rate for regularization.
	- l2_reg: L2 regularization parameter.
	- learning_rate: Learning rate for the optimizer.

	Returns:
	- model: Compiled Keras model.
	"""
	# Sequential allows for building on/adding layers
	model = Sequential()
	# Fully-Connected or Dense layer, all the neurons are connected to the next/previous layer
	model.add(Dense(units, input_dim=input_dim, activation=activation, kernel_regularizer=l2(l2_reg)))
	model.add(Dropout(dropout_rate))  # Dropout layer works by randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
	model.add(BatchNormalization())  # Batch normalization works by normalizing the input layer by adjusting and scaling the activations.
	
	"""
	•	L1 and L2 Regularization:
		•	L2 Regularization (Ridge): Adds a penalty equal to the sum of the squared weights to the loss function (e.g., Dense(units, kernel_regularizer=l2(0.01))).
		•	L1 Regularization (Lasso): Adds a penalty equal to the sum of the absolute values of the weights (e.g., Dense(units, kernel_regularizer=l1(0.01))).
		•	Elastic Net: Combines L1 and L2 regularization.
	"""
	for _ in range(layers - 1):
		model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))
		model.add(Dropout(dropout_rate))
		model.add(BatchNormalization())
	
	model.add(Dense(1))  # Output layer for regression
	model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
	
	return model


def get_model_memory_usage(batch_size, model):
	import numpy as np
	
	try:
		from keras import backend as K
	except:
		from tensorflow.keras import backend as K
	
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
	try:
		from keras import backend as K
	except:
		from tensorflow.keras import backend as K
	
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
	try:
		from keras import backend as K
	except:
		from tensorflow.keras import backend as K
	
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


# Example usage of hyperparameter tuning with RandomizedSearchCV
def hyperparameter_tuning(X_train, y_train, input_dim):
	model = KerasRegressor(build_fn=build_neural_network, input_dim=input_dim, verbose=0)
	
	param_dist = {
		'layers': [1, 2, 3],
		'units': [32, 64, 128],
		'activation': ['relu', 'tanh'],  # 'sigmoid'
		'dropout_rate': [0.2, 0.5, 0.7],
		'l2_reg': [0.01, 0.001, 0.0001],
		'learning_rate': [0.01, 0.001, 0.0001],
		'batch_size': [16, 32, 64, 128, 256],
		'epochs': [50, 100, 200]
	}
	
	random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=20, cv=3, verbose=2, scoring=make_scorer(mean_squared_error, greater_is_better=False))
	
	random_search_result = random_search.fit(X_train, y_train)
	
	# Create a DataFrame to store all results
	results_df = pd.DataFrame(random_search_result.cv_results_)
	
	# Extract relevant columns and sort by rank_test_score
	results_df = results_df[['params', 'mean_test_score', 'rank_test_score']]
	sorted_results = results_df.sort_values(by='rank_test_score')
	
	# Display all sorted results
	print("All Sorted Results:")
	print(sorted_results)
	
	# Display the best parameters and score
	print("\nBest Parameters:", random_search_result.best_params_)
	print("Best Score:", random_search_result.best_score_)
	
	return random_search_result.best_estimator_


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
			if isinstance(layer, tf.keras.layers.BatchNormalization):
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
