import torch
from sklearn.preprocessing import StandardScaler

from Configuration_Variables import *
from Supplemental_Sensor_Graph_Functions import *


def load_and_prepare_data(sensor_num, test_num, bit_resolution, mapping='ADC_vs_N'):
	# Load data for the current test
	instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=test_num))
	arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=test_num))
	
	min_length = min(len(instron_data), len(arduino_data))
	instron_force = instron_data["Force [N]"].iloc[:min_length].values.reshape(-1, 1)
	sensor_adc = arduino_data[f"ADC{sensor_num}"].iloc[:min_length].values.reshape(-1, 1)
	
	# Optional quantization step (if needed)
	instron_force_quantized = quantize_data(instron_force.flatten(), bit_resolution)
	sensor_adc_quantized = quantize_data(sensor_adc.flatten(), bit_resolution)
	
	# Depending on the mapping, set inputs and targets
	if mapping == 'ADC_vs_N':
		inputs = instron_force_quantized.reshape(-1, 1)  # Instron force as input
		targets = sensor_adc_quantized.reshape(-1, 1)  # Raw ADC values as targets
	elif mapping == 'N_vs_N':
		inputs = sensor_adc_quantized.reshape(-1, 1)  # ADC values as input
		targets = instron_force_quantized.reshape(-1, 1)  # Instron force as target
	else:
		raise ValueError("Invalid mapping type. Use 'ADC_vs_N' or 'N_vs_N'.")
	
	return inputs, targets, instron_force, sensor_adc


def train_model_with_hyperparameter_tuning(inputs, targets, bit_resolution, test_num, hyperparams_dict):
	from sklearn.model_selection import train_test_split
	import itertools
	
	# Unpack hyperparameter lists
	units_list = hyperparams_dict.get('units_list', [64, 128])
	layers_list = hyperparams_dict.get('layers_list', [1, 2])
	activation_list = hyperparams_dict.get('activation_list', ['tanh'])
	dropout_rate_list = hyperparams_dict.get('dropout_rate_list', [0.0, 0.1])
	l2_reg_list = hyperparams_dict.get('l2_reg_list', [0.0001, 0.001])
	learning_rate_list = hyperparams_dict.get('learning_rate_list', [0.0005, 0.001])
	epochs_list = hyperparams_dict.get('epochs_list', [100])
	batch_size_list = hyperparams_dict.get('batch_size_list', [64, 256])
	
	# Split the data into training and validation sets
	X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)
	
	# Scale the data
	input_scaler = StandardScaler()
	output_scaler = StandardScaler()
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
		print(f"Test {test_num}, Hyperparameters: units={units_}, layers={layers_}, activation={activation_}, dropout_rate={dropout_rate_}, l2_reg={l2_reg_}, learning_rate={learning_rate_}, epochs={epochs_}, batch_size={batch_size_}")
		
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
	
	# Retrain the best model on the full dataset
	model = QuantizedNN(
		input_dim=1, units=best_hyperparams['units'], layers=best_hyperparams['layers'],
		activation=best_hyperparams['activation'], dropout_rate=best_hyperparams['dropout_rate'],
		weight_bit_width=bit_resolution, act_bit_width=bit_resolution
	)
	model.load_state_dict(best_model_state)
	return model, input_scaler, output_scaler, best_hyperparams


def train_model(inputs, targets, units, layers, activation, dropout_rate, l2_reg, learning_rate, epochs, batch_size, bit_resolution):
	# Initialize scalers
	input_scaler = StandardScaler()
	output_scaler = StandardScaler()
	inputs_scaled = input_scaler.fit_transform(inputs)
	targets_scaled = output_scaler.fit_transform(targets)
	
	# Initialize and train the quantized neural network
	model = QuantizedNN(
		input_dim=1, units=units, layers=layers, activation=activation, dropout_rate=dropout_rate,
		weight_bit_width=bit_resolution, act_bit_width=bit_resolution
	)
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
	
	# Convert data to PyTorch tensors for training
	inputs_tensor = torch.Tensor(inputs_scaled).to(device)
	targets_tensor = torch.Tensor(targets_scaled).to(device)
	
	dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
	
	# Train the model
	for epoch in range(epochs):
		model.train()
		total_loss = 0
		for x_batch, y_batch in dataloader:
			optimizer.zero_grad()
			outputs = model(x_batch)
			loss = criterion(outputs, y_batch)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		if (epoch + 1) % 10 == 0:
			print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.6f}")
	
	return model, input_scaler, output_scaler


def evaluate_model(model, inputs, instron_force, sensor_adc, input_scaler, output_scaler, mapping):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.eval()
	with torch.no_grad():
		inputs_scaled = input_scaler.transform(inputs)
		inputs_tensor = torch.Tensor(inputs_scaled).to(device)
		outputs_scaled = model(inputs_tensor).cpu().numpy()
		outputs = output_scaler.inverse_transform(outputs_scaled)
	
	# First Graph: Residuals in N (calibrated sensor N - Instron N)
	if mapping == 'N_vs_N':
		# Residuals are calculated as the difference between the calibrated N and Instron N
		residuals = outputs.flatten() - instron_force.flatten()
		return outputs.flatten(), residuals
	
	# Second Graph: Residuals in ADC (Instron N - ADC values)
	elif mapping == 'ADC_vs_N':
		# Residuals are the difference between the ADC values and the predicted output from the model
		residuals = sensor_adc.flatten() - outputs.flatten()
		return outputs, residuals


def plot_overlay(overlay_ax, inputs, targets, outputs, test_num, mapping):
	if mapping == 'N_vs_N':
		# Plot calibrated N values (Y) vs Instron N (X)
		x = targets.flatten()  # Instron N
		y_pred = outputs.flatten()  # Predicted N (Calibrated)
		xlabel = "Instron Force [N]"
		ylabel = "Calibrated Force [N]"
	elif mapping == 'ADC_vs_N':
		# Plot ADC vs Instron N
		x = targets.flatten()  # Instron N
		y_pred = outputs.flatten()  # Predicted ADC
		xlabel = "Instron Force [N]"
		ylabel = "ADC Value"
	else:
		raise ValueError("Invalid mapping type. Use 'ADC_vs_N' or 'N_vs_N'.")
	
	overlay_ax.plot(x, y_pred, label=f"Test {test_num} - Neural Fit", linewidth=2)
	overlay_ax.set_xlabel(xlabel)
	overlay_ax.set_ylabel(ylabel)
	overlay_ax.grid(True)


def plot_residuals(residuals_ax, instron_force, residuals, test_num, mapping):
	x = instron_force.flatten()  # Instron Force (N) is the baseline
	
	# Invert the x-axis to make the direction go from larger to smaller
	residuals_ax.invert_xaxis()
	
	if mapping == 'N_vs_N':
		# First graph: Residuals in N vs Instron N
		residuals_ax.plot(x, residuals, label=f"Residuals in N (Test {test_num})", linewidth=2)
		residuals_ax.set_xlabel("Instron Force [N]")
		residuals_ax.set_ylabel("Residuals in N")
	elif mapping == 'ADC_vs_N':
		# Second graph: Residuals in ADC vs Instron N
		residuals_ax.plot(x, residuals, label=f"Residuals in ADC (Test {test_num})", linewidth=2)
		residuals_ax.set_xlabel("Instron Force [N]")
		residuals_ax.set_ylabel("Residuals in ADC")
	else:
		raise ValueError("Invalid mapping type. Use 'N_vs_N' or 'ADC_vs_N'.")
	residuals_ax.grid(True)
