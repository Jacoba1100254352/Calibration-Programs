import random

import brevitas.nn as qnn
import torch
import torch.nn as nn
from brevitas.core.scaling import ScalingImplType
from brevitas.quant import Int8WeightPerTensorFixedPoint
from keras.api.layers import BatchNormalization, Dense, Dropout
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from keras.api.regularizers import l2
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Workflow_Programs.Configuration_Variables import *
from Workflow_Programs.Supplemental_Sensor_Graph_Functions import *


seed_value = 42


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
			'sigmoid': qnn.QuantSigmoid,
			'hardtanh': qnn.QuantHardTanh,
			'identity': qnn.QuantIdentity
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


def calculate_linear_fit(instron_force, arduino_raw_force):
	"""
	Calculate the linear regression coefficients using the closed-form formula
	for computational efficiency.

	:param instron_force: Array-like, force data from Excel (Instron data).
	:param arduino_raw_force: Array-like, raw force data from Arduino.
	:return: Tuple, (m, b) coefficients from the linear fit.
	"""
	# Convert input to NumPy arrays for efficient calculation
	x = np.array(arduino_raw_force)
	y = np.array(instron_force)
	
	# Compute sums and necessary terms for the closed-form solution
	n = len(x)
	sum_x = np.sum(x)
	sum_y = np.sum(y)
	sum_x_squared = np.sum(x * x)
	sum_xy = np.sum(x * y)
	
	# Calculate slope (m) and intercept (b) using the closed-form formulas
	m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
	b = (sum_y - m * sum_x) / n
	
	return m, b


def load_and_prepare_data_with_linear(sensor_num, test_num, bit_resolution):
	# Load data for the current test
	instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=test_num))
	arduino_data = pd.read_csv(get_data_filepath(ALIGNED_ARDUINO_DIR, sensor_num, _TEST_NUM=test_num))
	
	min_length = min(len(instron_data), len(arduino_data))
	instron_N_unquantized = instron_data["Force [N]"].iloc[:min_length]
	sensor_adc_unquantized = arduino_data[f"ADC{sensor_num}"].iloc[:min_length]
	
	instron_N_unquantized = instron_N_unquantized.values.reshape(-1, 1)
	sensor_adc_unquantized = sensor_adc_unquantized.values.reshape(-1, 1)
	
	# Optional quantization step (if needed)
	instron_N_quantized = -quantize_data(instron_N_unquantized.flatten(), bit_resolution)
	sensor_adc_quantized = -quantize_data(sensor_adc_unquantized.flatten(), bit_resolution)
	
	# Depending on the mapping, set inputs and targets
	inputs = sensor_adc_quantized.reshape(-1, 1)  # ADC values as input
	targets = instron_N_quantized.reshape(-1, 1)  # Instron force as target
	
	return inputs, targets, -instron_N_unquantized, -sensor_adc_unquantized


def train_model_with_hyperparameter_tuning(inputs, targets, bit_resolution, test_num, hyperparams_dict):
	from sklearn.model_selection import train_test_split
	import itertools
	
	torch.manual_seed(seed_value)
	np.random.seed(seed_value)
	random.seed(seed_value)
	
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed_value)
		torch.cuda.manual_seed_all(seed_value)
	
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	# Unpack hyperparameter lists
	# units_list = hyperparams_dict.get('units_list', [16, 32, 64])
	# layers_list = hyperparams_dict.get('layers_list', [1])
	# activation_list = hyperparams_dict.get('activation_list', ['relu'])
	# dropout_rate_list = hyperparams_dict.get('dropout_rate_list', [0.15, 0.2, 0.25])
	# l2_reg_list = hyperparams_dict.get('l2_reg_list', [0.0025, 0.005, 0.0075])
	# learning_rate_list = hyperparams_dict.get('learning_rate_list', [0.0001, 0.00025, 0.0005])
	# epochs_list = hyperparams_dict.get('epochs_list', [100])
	# batch_size_list = hyperparams_dict.get('batch_size_list', [8, 16, 32])
	units_list = hyperparams_dict.get('units_list', [32, 64, 128, 256])
	layers_list = hyperparams_dict.get('layers_list', [1, 2])
	activation_list = hyperparams_dict.get('activation_list', ['relu'])
	dropout_rate_list = hyperparams_dict.get('dropout_rate_list', [0.0, 0.1, 0.2, 0.5])
	l2_reg_list = hyperparams_dict.get('l2_reg_list', [0.005, 0.01, 0.02])
	learning_rate_list = hyperparams_dict.get('learning_rate_list', [0.0005, 0.001, 0.002])  # Try 0.0001 or 0.00001
	epochs_list = hyperparams_dict.get('epochs_list', [100])
	batch_size_list = hyperparams_dict.get('batch_size_list', [32, 64, 256])
	
	# Split the data into training and validation sets
	X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.2, random_state=seed_value)
	
	# Scale the data
	input_scaler = StandardScaler()
	output_scaler = StandardScaler()
	# Reshape the targets to be 2D (needed by the scaler)
	X_train = X_train.reshape(-1, 1)
	y_train = y_train.reshape(-1, 1)
	y_val = y_val.reshape(-1, 1)
	X_train_scaled = input_scaler.fit_transform(X_train)
	y_train_scaled = output_scaler.fit_transform(y_train)
	X_val_scaled = input_scaler.transform(X_val)
	y_val_scaled = output_scaler.transform(y_val)
	
	# Define the hyperparameter grid
	hyperparameter_grid = list(itertools.product(
		units_list, layers_list, activation_list,
		dropout_rate_list, l2_reg_list, learning_rate_list,
		epochs_list, batch_size_list
	))
	
	best_val_loss = float('inf')
	best_hyperparams = None
	best_model_state = None
	
	results_list = []
	
	# Hyperparameter tuning
	for (units_, layers_, activation_, dropout_rate_, l2_reg_, learning_rate_, epochs_, batch_size_) in hyperparameter_grid:
		print(f"Test {test_num}, Hyperparameters: units={units_}, layers={layers_}, activation={activation_}, dropout_rate={dropout_rate_}, l2_reg={l2_reg_}, learning_rate={learning_rate_}, epochs={epochs_}, batch_size={batch_size_}", end="")
		
		# Initialize the model
		model = QuantizedNN(
			input_dim=1, units=units_, layers=layers_, activation=activation_, dropout_rate=dropout_rate_,
			weight_bit_width=bit_resolution, act_bit_width=bit_resolution
		)
		
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model.to(device)
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_, weight_decay=l2_reg_)
		
		# Convert data to PyTorch tensors for training
		X_train_tensor = torch.Tensor(X_train_scaled).to(device)
		y_train_tensor = torch.Tensor(y_train_scaled).to(device)
		X_val_tensor = torch.Tensor(X_val_scaled).to(device)
		y_val_tensor = torch.Tensor(y_val_scaled).to(device)
		
		train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_, shuffle=True)
		
		# Train the model
		for epoch in range(epochs_):
			model.train()
			total_loss = 0
			for x_batch, y_batch in train_dataloader:
				optimizer.zero_grad()
				outputs = model(x_batch)
				loss = criterion(outputs, y_batch)
				loss.backward()
				optimizer.step()
				total_loss += loss.item()
		
		# Validation
		model.eval()
		with torch.no_grad():
			val_outputs = model(X_val_tensor)
			val_loss = criterion(val_outputs, y_val_tensor).item()
		
		# Save results
		results_list.append({
			'units': units_,
			'layers': layers_,
			'activation': activation_,
			'dropout_rate': dropout_rate_,
			'l2_reg': l2_reg_,
			'learning_rate': learning_rate_,
			'epochs': epochs_,
			'batch_size': batch_size_,
			'val_loss': val_loss
		})
		
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_hyperparams = {
				'units': units_,
				'layers': layers_,
				'activation': activation_,
				'dropout_rate': dropout_rate_,
				'l2_reg': l2_reg_,
				'learning_rate': learning_rate_,
				'epochs': epochs_,
				'batch_size': batch_size_
			}
			best_model_state = model.state_dict()
		
		print(f", Validation Loss: {val_loss:.6f}")
	
	print(f"Best Validation Loss: {best_val_loss:.6f}")
	print(f"Best Hyperparameters: {best_hyperparams}")
	print(f"Best Model State: {best_model_state}")
	
	# Retrain the best model on the full dataset
	model = QuantizedNN(
		input_dim=1, units=best_hyperparams['units'], layers=best_hyperparams['layers'],
		activation=best_hyperparams['activation'], dropout_rate=best_hyperparams['dropout_rate'],
		weight_bit_width=bit_resolution, act_bit_width=bit_resolution
	)
	model.load_state_dict(best_model_state)
	return model, input_scaler, output_scaler, best_hyperparams


def train_model(
	inputs, residuals_input, units, layers, activation, dropout_rate,
	l2_reg, learning_rate, epochs, batch_size, bit_resolution,
	validation_split=0.2, patience=20
):
	# Set random seeds for reproducibility
	torch.manual_seed(seed_value)
	np.random.seed(seed_value)
	random.seed(seed_value)
	
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed_value)
		torch.cuda.manual_seed_all(seed_value)
	
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	# Initialize scalers
	input_scaler = StandardScaler()
	output_scaler = MinMaxScaler(feature_range=(-1, 1))  # Scale residuals to [-1, 1]
	
	# Reshape the inputs and residuals to be 2D (needed by the scaler)
	inputs = inputs.reshape(-1, 1)
	residuals_input = residuals_input.reshape(-1, 1)
	
	# Split data into training and validation sets
	inputs_train, inputs_val, residuals_train, residuals_val = train_test_split(
		inputs, residuals_input, test_size=validation_split, random_state=seed_value
	)
	
	# Scale the data
	inputs_train_scaled = input_scaler.fit_transform(inputs_train)
	inputs_val_scaled = input_scaler.transform(inputs_val)
	
	residuals_train_scaled = output_scaler.fit_transform(residuals_train)
	residuals_val_scaled = output_scaler.transform(residuals_val)
	
	# Initialize and train the quantized neural network
	model = QuantizedNN(
		input_dim=1, units=units, layers=layers, activation=activation,
		dropout_rate=dropout_rate,
		weight_bit_width=bit_resolution, act_bit_width=bit_resolution
	)
	
	# Initialize the model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
	
	# Initialize the learning rate scheduler
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
	
	# Early stopping parameters
	best_val_loss = float('inf')
	patience_counter = 0
	best_model_path = 'best_model.pth'
	
	# Convert data to PyTorch tensors for training and validation
	inputs_train_tensor = torch.Tensor(inputs_train_scaled).to(device)
	residuals_train_tensor = torch.Tensor(residuals_train_scaled).to(device)
	
	inputs_val_tensor = torch.Tensor(inputs_val_scaled).to(device)
	residuals_val_tensor = torch.Tensor(residuals_val_scaled).to(device)
	
	# Create DataLoaders
	train_dataset = torch.utils.data.TensorDataset(inputs_train_tensor, residuals_train_tensor)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	
	val_dataset = torch.utils.data.TensorDataset(inputs_val_tensor, residuals_val_tensor)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	
	# Training loop with early stopping
	for epoch in range(1, epochs + 1):
		model.train()
		train_loss = 0.0
		for x_batch, y_batch in train_loader:
			optimizer.zero_grad()
			outputs = model(x_batch)
			loss = criterion(outputs, y_batch)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		
		# Calculate average training loss
		avg_train_loss = train_loss / len(train_loader)
		
		# Evaluate on validation set
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for x_val, y_val in val_loader:
				outputs_val = model(x_val)
				loss_val = criterion(outputs_val, y_val)
				val_loss += loss_val.item()
		avg_val_loss = val_loss / len(val_loader)
		
		# Step the scheduler based on validation loss
		scheduler.step(avg_val_loss)
		
		# Early stopping check
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			patience_counter = 0
			torch.save(model.state_dict(), best_model_path)
		# Uncomment the next line for detailed logs
		# print(f"Epoch {epoch}: Validation loss improved to {avg_val_loss:.6f}. Saving model.")
		else:
			patience_counter += 1
			# Uncomment the next line for detailed logs
			# print(f"Epoch {epoch}: Validation loss did not improve. Patience counter: {patience_counter}")
			if patience_counter >= patience:
				print("Early stopping triggered.")
				break
		
		# Print progress every 10 epochs
		if epoch % 10 == 0 or epoch == 1:
			print(f"Epoch [{epoch}/{epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
	
	# Load the best model
	model.load_state_dict(torch.load(best_model_path))
	
	return model, input_scaler, output_scaler


def evaluate_model(model, inputs, residuals_input, input_scaler, output_scaler):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.eval()
	with torch.no_grad():
		inputs_scaled = input_scaler.transform(inputs)
		inputs_tensor = torch.Tensor(inputs_scaled).to(device)
		outputs_scaled = model(inputs_tensor).cpu().numpy()
		residual_corrections = output_scaler.inverse_transform(outputs_scaled).flatten()
	
	return residual_corrections


# Helper function for quantizing data (if using custom bit resolutions for inputs/outputs)
def quantize_data(data, bit_resolution):
	"""Quantize the data to the given bit resolution."""
	max_val = np.max(np.abs(data))
	scale = (2**bit_resolution) - 1
	quantized_data = np.round(data / max_val * scale) * max_val / scale
	return quantized_data


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
