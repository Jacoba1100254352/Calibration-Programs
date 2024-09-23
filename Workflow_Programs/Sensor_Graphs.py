import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler

from Configuration_Variables import *
from Supplemental_Sensor_Graph_Functions import *


def analyze_and_graph_neural_fit_single_pdf_combined_multiple_tests(
	test_range, sensor_num, units=64, layers=2, activation='relu', dropout_rate=0.5, l2_reg=0.01,
	learning_rate=0.001, epochs=100, batch_size=32, window_size=None, poly_order=None,
	smoothing_method="boxcar", save_graphs=True, show_graphs=True, bit_resolution=12
):
	"""
	Analyze and visualize neural network fits for each test and compare them across multiple tests.
	This function fits a neural network model to each test individually, graphs the combined ADC vs Instron Force (N) data,
	subtracts the individual neural fit from the combined data, and graphs the residuals for each test on the same graph.
	"""
	
	# Initialize scalers
	input_scaler = StandardScaler()
	output_scaler = StandardScaler()
	
	# Initialize lists for storing combined data and residuals for each test
	combined_instron_force = []
	combined_arduino_force = []
	
	# Initialize the PDF to save the graphs
	with PdfPages(f"/Users/jacobanderson/Downloads/Neural_Network_Fit_Sensor_Set_{sensor_num}.pdf") as pdf:
		# Set up figures for overlay and residuals
		plt.figure(figsize=(10, 6))
		overlay_fig = plt.figure(figsize=(10, 6))
		residuals_fig = plt.figure(figsize=(10, 6))
		
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
			
			# Scale the data
			instron_force_scaled = input_scaler.fit_transform(instron_force_quantized.reshape(-1, 1))
			updated_arduino_force_scaled = output_scaler.fit_transform(updated_arduino_force_quantized.reshape(-1, 1))
			
			combined_instron_force.append(instron_force_scaled)
			combined_arduino_force.append(updated_arduino_force_scaled)
			
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
			
			# Plot the overlay: Combined data and neural fit for this test
			plt.figure(overlay_fig.number)
			plt.plot(instron_force_orig, arduino_force_orig, label=f"Test {_TEST_NUM} - Data", linestyle='--', linewidth=1)
			plt.plot(instron_force_orig, combined_outputs, label=f"Test {_TEST_NUM} - Neural Fit", linewidth=2)
			
			# Calculate residuals: ADC - Neural Fit
			residuals = arduino_force_orig.flatten() - combined_outputs.flatten()
			residuals_smoothed = apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
			
			# Ensure Instron force and residuals_smoothed have the same length
			if len(instron_force_orig) != len(residuals_smoothed):
				residuals_smoothed = residuals_smoothed[:len(instron_force_orig)]
			
			# Plot the residuals for this test
			plt.figure(residuals_fig.number)
			plt.plot(instron_force_orig, residuals_smoothed, label=f"Residuals (Test {_TEST_NUM})", linewidth=2)
		
		# Finalize overlay plot
		plt.figure(overlay_fig.number)
		plt.xlabel("Instron Force [N]")
		plt.ylabel("ADC Value")
		plt.title(f"Overlay of ADC vs Instron Force and Neural Fit for Tests {test_range}")
		plt.legend(loc="upper left")
		plt.grid(True)
		if show_graphs:
			plt.show()
		if save_graphs:
			pdf.savefig()
		plt.close()
		
		# Finalize residuals plot
		plt.figure(residuals_fig.number)
		plt.xlabel("Instron Force [N]")
		plt.ylabel("Residuals (ADC - Neural Fit)")
		plt.title(f"Residuals for Tests {test_range}")
		plt.legend(loc="lower left")
		plt.grid(True)
		plt.gca().invert_xaxis()
		if show_graphs:
			plt.show()
		if save_graphs:
			pdf.savefig()
		plt.close()


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
