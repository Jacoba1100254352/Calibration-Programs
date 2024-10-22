import random

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Configuration_Variables import *
from Supplemental_Sensor_Graph_Functions import *


seed_value = 42


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
