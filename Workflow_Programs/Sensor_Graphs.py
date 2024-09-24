import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler

from Configuration_Variables import *
from Supplemental_Sensor_Graph_Functions import *


def analyze_and_graph_neural_fit_single_pdf_combined_multiple_tests(
	test_range, sensor_num, units=64, layers=2, activation='relu', dropout_rate=0.5, l2_reg=0.01,
	learning_rate=0.001, epochs=100, batch_size=32, window_size=None, poly_order=None,
	smoothing_method="boxcar", save_graphs=True, show_graphs=True, bit_resolution=12,
	enable_hyperparameter_tuning=False,
	units_list=None, layers_list=None, activation_list=None,
	dropout_rate_list=None, l2_reg_list=None, learning_rate_list=None,
	bit_resolution_list=None, epochs_list=None, batch_size_list=None
):
	"""
	Analyze and visualize neural network fits for each test and compare them across multiple tests.
	This function fits a neural network model to each test individually, graphs the combined ADC vs Instron Force (N) data,
	subtracts the individual neural fit from the combined data, and graphs the residuals for each test on the same graph.
	If enable_hyperparameter_tuning is True, performs hyperparameter tuning over specified ranges.
	"""
	
	import matplotlib.pyplot as plt
	import itertools
	from sklearn.model_selection import train_test_split
	
	# Close all existing figures
	plt.close('all')
	
	# Initialize the PDF to save the graphs
	with PdfPages(f"/Users/jacobanderson/Downloads/Neural_Network_Fit_Sensor_Set_{sensor_num}.pdf") as pdf:
		# Set up figures and axes for overlay and residuals
		overlay_fig, overlay_ax = plt.subplots(figsize=(10, 6))
		residuals_fig, residuals_ax = plt.subplots(figsize=(10, 6))
		
		# Loop over each test in the range and perform neural fit separately
		for _TEST_NUM in test_range:
			# Load data for the current test
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			
			min_length = min(len(instron_data), len(updated_arduino_data))
			instron_force = instron_data["Force [N]"].iloc[:min_length].values.reshape(-1, 1)
			updated_arduino_force = updated_arduino_data[f"ADC{sensor_num}"].iloc[:min_length].values.reshape(-1, 1)
			
			# Optional quantization step (if needed)
			instron_force_quantized = quantize_data(instron_force.flatten(), bit_resolution)
			updated_arduino_force_quantized = quantize_data(updated_arduino_force.flatten(), bit_resolution)
			
			if enable_hyperparameter_tuning:
				# Split the data into training and validation sets
				X_train, X_val, y_train, y_val = train_test_split(
					instron_force_quantized.reshape(-1, 1),
					updated_arduino_force_quantized.reshape(-1, 1),
					test_size=0.2, random_state=42
				)
				
				# Scale the data
				input_scaler = StandardScaler()
				output_scaler = StandardScaler()
				X_train_scaled = input_scaler.fit_transform(X_train)
				y_train_scaled = output_scaler.fit_transform(y_train)
				X_val_scaled = input_scaler.transform(X_val)
				y_val_scaled = output_scaler.transform(y_val)
				
				# Define default hyperparameter ranges if not provided
				# if units_list is None:
				# 	units_list = [64, 160]
				# if layers_list is None:
				# 	layers_list = [1]
				# if activation_list is None:
				# 	activation_list = ['tanh']
				# if dropout_rate_list is None:
				# 	dropout_rate_list = [0.1, 0.2]
				# if l2_reg_list is None:
				# 	l2_reg_list = [0.01, 0.02, 0.05]
				# if learning_rate_list is None:
				# 	learning_rate_list = [0.0005, 0.001, 0.002]
				# if bit_resolution_list is None:
				# 	bit_resolution_list = [8, 14]
				# if epochs_list is None:
				# 	epochs_list = [50, 200]
				# if batch_size_list is None:
				# 	batch_size_list = [32, 256]
				
				if units_list is None:
					units_list = [64, 96, 128, 160]
				if layers_list is None:
					layers_list = [1]
				if activation_list is None:
					activation_list = ['tanh']
				if dropout_rate_list is None:
					dropout_rate_list = [0.0, 0.1, 0.2]
				if l2_reg_list is None:
					l2_reg_list = [0.005, 0.01, 0.02]
				if learning_rate_list is None:
					learning_rate_list = [0.0005, 0.00075, 0.001]
				if bit_resolution_list is None:
					bit_resolution_list = [8, 10, 12]
				if epochs_list is None:
					epochs_list = [100]
				if batch_size_list is None:
					batch_size_list = [32, 64]
				
				# Define the hyperparameter grid
				hyperparameter_grid = list(itertools.product(
					units_list, layers_list, activation_list,
					dropout_rate_list, l2_reg_list, learning_rate_list,
					bit_resolution_list, epochs_list, batch_size_list
				))
				
				best_val_loss = float('inf')
				best_hyperparams = None
				best_model_state = None
				
				# Initialize list to keep track of results
				results_list = []
				
				# Hyperparameter tuning
				for (units_, layers_, activation_, dropout_rate_, l2_reg_, learning_rate_, bit_resolution_, epochs_, batch_size_) in hyperparameter_grid:
					tmpObject = {
						'units': units_,
						'layers': layers_,
						'activation': activation_,
						'dropout_rate': dropout_rate_,
						'l2_reg': l2_reg_,
						'learning_rate': learning_rate_,
						'bit_resolution': bit_resolution_,
						'epochs': epochs_,
						'batch_size': batch_size_
					}
					print(f"Test {_TEST_NUM}, Current hyperparameters: {tmpObject}", end="")
					
					# Initialize the model with current hyperparameters
					model = QuantizedNN(
						input_dim=1, units=units_, layers=layers_, activation=activation_, dropout_rate=dropout_rate_,
						weight_bit_width=bit_resolution_, act_bit_width=bit_resolution_
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
						num_batches = 0
						for x_batch, y_batch in train_dataloader:
							optimizer.zero_grad()
							outputs = model(x_batch)
							loss = criterion(outputs, y_batch)
							loss.backward()
							optimizer.step()
							total_loss += loss.item()
							num_batches += 1
					
					# Validation
					model.eval()
					with torch.no_grad():
						val_outputs = model(X_val_tensor)
						val_loss = criterion(val_outputs, y_val_tensor).item()
					
					# Append results to the list
					results_list.append({
						'units': units_,
						'layers': layers_,
						'activation': activation_,
						'dropout_rate': dropout_rate_,
						'l2_reg': l2_reg_,
						'learning_rate': learning_rate_,
						'bit_resolution': bit_resolution_,
						'epochs': epochs_,
						'batch_size': batch_size_,
						'val_loss': val_loss
					})
					
					print(", val_loss:", val_loss)
					
					# If this model is better, save its parameters and hyperparameters
					if val_loss < best_val_loss:
						best_val_loss = val_loss
						best_hyperparams = {
							'units': units_,
							'layers': layers_,
							'activation': activation_,
							'dropout_rate': dropout_rate_,
							'l2_reg': l2_reg_,
							'learning_rate': learning_rate_,
							'bit_resolution': bit_resolution_,
							'epochs': epochs_,
							'batch_size': batch_size_
						}
						best_model_state = model.state_dict()
				
				# Sort the results list based on validation loss
				results_list.sort(key=lambda x: x['val_loss'])
				
				# Print out the best hyperparameters and validation loss
				print(f"\nBest hyperparameters for Test {_TEST_NUM}:")
				print(best_hyperparams)
				print(f"Best validation loss: {best_val_loss}")
				print(f"Best Model State: {best_model_state}")
				
				# Print out all hyperparameter combinations and their validation losses
				print(f"\nAll hyperparameter combinations and their validation losses for Test {_TEST_NUM}:")
				for result in results_list:
					print(result)
				
				# After hyperparameter tuning, retrain the best model on the full dataset
				# Re-initialize scalers on full dataset
				input_scaler_full = StandardScaler()
				output_scaler_full = StandardScaler()
				instron_force_scaled = input_scaler_full.fit_transform(instron_force_quantized.reshape(-1, 1))
				updated_arduino_force_scaled = output_scaler_full.fit_transform(updated_arduino_force_quantized.reshape(-1, 1))
				
				# Re-initialize the model with the best hyperparameters
				best_model = QuantizedNN(
					input_dim=1, units=best_hyperparams['units'], layers=best_hyperparams['layers'],
					activation=best_hyperparams['activation'], dropout_rate=best_hyperparams['dropout_rate'],
					weight_bit_width=bit_resolution, act_bit_width=bit_resolution
				)
				best_model.to(device)
				criterion = nn.MSELoss()
				optimizer = torch.optim.Adam(best_model.parameters(), lr=best_hyperparams['learning_rate'],
				                             weight_decay=best_hyperparams['l2_reg'])
				
				# Convert data to PyTorch tensors for training
				instron_force_tensor = torch.Tensor(instron_force_scaled).to(device)
				updated_arduino_force_tensor = torch.Tensor(updated_arduino_force_scaled).to(device)
				
				dataset_full = torch.utils.data.TensorDataset(instron_force_tensor, updated_arduino_force_tensor)
				dataloader_full = torch.utils.data.DataLoader(dataset_full, batch_size=batch_size, shuffle=True)
				
				# Retrain the model on the full dataset
				for epoch in range(epochs):
					best_model.train()
					total_loss = 0
					num_batches = 0
					for x_batch, y_batch in dataloader_full:
						optimizer.zero_grad()
						outputs = best_model(x_batch)
						loss = criterion(outputs, y_batch)
						loss.backward()
						optimizer.step()
						total_loss += loss.item()
						num_batches += 1
					if (epoch + 1) % 10 == 0:
						print(f"Test {_TEST_NUM} - Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / num_batches:.6f}")
				
				# After training, evaluate the model and calculate residuals
				best_model.eval()
				with torch.no_grad():
					combined_instron_tensor = torch.Tensor(instron_force_scaled).to(device)
					combined_outputs_scaled = best_model(combined_instron_tensor).cpu().numpy()
					combined_outputs = output_scaler_full.inverse_transform(combined_outputs_scaled)  # Inverse transform
				
				# Convert Instron force and Arduino force back to the original scale
				instron_force_orig = input_scaler_full.inverse_transform(instron_force_scaled)
				arduino_force_orig = output_scaler_full.inverse_transform(updated_arduino_force_scaled)
			
			else:
				# Without hyperparameter tuning, proceed as before
				# Initialize scalers
				input_scaler = StandardScaler()
				output_scaler = StandardScaler()
				instron_force_scaled = input_scaler.fit_transform(instron_force_quantized.reshape(-1, 1))
				updated_arduino_force_scaled = output_scaler.fit_transform(updated_arduino_force_quantized.reshape(-1, 1))
				
				# Initialize and train the quantized neural network for this test
				model = QuantizedNN(
					input_dim=1, units=units, layers=layers, activation=activation, dropout_rate=dropout_rate,
					weight_bit_width=bit_resolution, act_bit_width=bit_resolution
				)
				
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
				model.to(device)
				criterion = nn.MSELoss()
				optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
				
				# Convert data to PyTorch tensors for training
				instron_force_tensor = torch.Tensor(instron_force_scaled).to(device)
				updated_arduino_force_tensor = torch.Tensor(updated_arduino_force_scaled).to(device)
				
				dataset = torch.utils.data.TensorDataset(instron_force_tensor, updated_arduino_force_tensor)
				dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
				
				# Train the model
				for epoch in range(epochs):
					model.train()
					total_loss = 0
					num_batches = 0
					for x_batch, y_batch in dataloader:
						optimizer.zero_grad()
						outputs = model(x_batch)
						loss = criterion(outputs, y_batch)
						loss.backward()
						optimizer.step()
						total_loss += loss.item()
						num_batches += 1
					if (epoch + 1) % 10 == 0:
						print(f"Test {_TEST_NUM} - Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / num_batches:.6f}")
				
				# After training, evaluate the model and calculate residuals
				model.eval()
				with torch.no_grad():
					combined_instron_tensor = torch.Tensor(instron_force_scaled).to(device)
					combined_outputs_scaled = model(combined_instron_tensor).cpu().numpy()
					combined_outputs = output_scaler.inverse_transform(combined_outputs_scaled)  # Inverse transform
				
				# Convert Instron force and Arduino force back to the original scale
				instron_force_orig = input_scaler.inverse_transform(instron_force_scaled)
				arduino_force_orig = output_scaler.inverse_transform(updated_arduino_force_scaled)
			
			# The rest of the code is common to both cases
			# Plot the overlay: Combined data and neural fit for this test
			overlay_ax.plot(instron_force_orig.flatten(), arduino_force_orig.flatten(),
			                label=f"Test {_TEST_NUM} - Data", linestyle='--', linewidth=1)
			overlay_ax.plot(instron_force_orig.flatten(), combined_outputs.flatten(),
			                label=f"Test {_TEST_NUM} - Neural Fit", linewidth=2)
			
			# Calculate residuals: ADC - Neural Fit
			residuals = arduino_force_orig.flatten() - combined_outputs.flatten()
			residuals_smoothed = apply_smoothing(residuals, method=smoothing_method,
			                                     window_size=window_size, poly_order=poly_order)
			
			# Ensure Instron force and residuals_smoothed have the same length
			if len(instron_force_orig) != len(residuals_smoothed):
				residuals_smoothed = residuals_smoothed[:len(instron_force_orig)]
			
			# Plot the residuals for this test
			residuals_ax.plot(instron_force_orig.flatten(), residuals_smoothed,
			                  label=f"Residuals (Test {_TEST_NUM})", linewidth=2)
		
		# Finalize overlay plot
		overlay_ax.set_xlabel("Instron Force [N]")
		overlay_ax.set_ylabel("ADC Value")
		overlay_ax.set_title(f"Overlay of ADC vs Instron Force and Neural Fit for Tests {test_range}")
		overlay_ax.grid(True)
		overlay_fig.tight_layout()
		
		# Collect handles and labels for legend
		handles, labels = overlay_ax.get_legend_handles_labels()
		overlay_ax.legend(handles, labels, loc="upper left")
		
		if save_graphs:
			pdf.savefig(overlay_fig)
		
		# Finalize residuals plot
		residuals_ax.set_xlabel("Instron Force [N]")
		residuals_ax.set_ylabel("Residuals (ADC - Neural Fit)")
		residuals_ax.set_title(f"Residuals for Tests {test_range}")
		residuals_ax.grid(True)
		residuals_ax.invert_xaxis()
		residuals_fig.tight_layout()
		
		# Collect handles and labels for legend
		handles, labels = residuals_ax.get_legend_handles_labels()
		residuals_ax.legend(handles, labels, loc="lower left")
		
		if save_graphs:
			pdf.savefig(residuals_fig)
		
		if show_graphs:
			plt.show()
		plt.close(overlay_fig)
		plt.close(residuals_fig)


def analyze_and_graph_calibrated_data_and_fits_single_pdf_combined_multiple_tests(
	test_range, sensor_num, window_size=None, poly_order=2, smoothing_method=None, save_graphs=True, show_graphs=True, bit_resolution=12
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
	"""
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
				
				# Use this for meta-analysis of calibration
				# updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"].iloc[:min_length]
				
				# Use this for analysis of raw ADC with instron N
				updated_arduino_force = updated_arduino_data["ADC" if SIMPLIFY else f"ADC{sensor_num}"].iloc[:min_length]
				
				# Quantize input and output data
				instron_force = quantize_data(instron_force, bit_resolution)
				updated_arduino_force = quantize_data(updated_arduino_force, bit_resolution)
				
				# Fit the polynomial model
				lin_fit = calculate_line_of_best_fit(x=instron_force, y=updated_arduino_force, isPolyfit=True, order=order)
				residuals = updated_arduino_force - lin_fit
				
				# Apply smoothing using the specified method
				residuals_smoothed = apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
				
				# Plot the smoothed residuals
				plt.plot(instron_force[:len(residuals_smoothed)], residuals_smoothed, '-', label=f"Test {_TEST_NUM}", linewidth=2)
			
			plt.xlabel("Instron Force [N]")
			plt.ylabel("Residual Force [ADC]")  # N
			plt.legend(loc="lower left")
			plt.title(f"Residuals for Polynomial Fit (Order {order}) Across Multiple Tests")
			plt.grid(True)
			plt.gca().invert_xaxis()
			
			if show_graphs:
				plt.show()
			
			if save_graphs:
				pdf.savefig()
			
			plt.close()
