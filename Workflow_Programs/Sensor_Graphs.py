import torch
from matplotlib.backends.backend_pdf import PdfPages

from Configuration_Variables import *
from Supplemental_Sensor_Graph_Functions import *


def analyze_and_graph_neural_fit_single_pdf_combined_multiple_tests(
	test_range, sensor_num, units=64, layers=2, activation='relu', dropout_rate=0.5, l2_reg=0.01,
	learning_rate=0.001, epochs=100, batch_size=32, window_size=None, poly_order=None,
	smoothing_method="boxcar", save_graphs=True, show_graphs=True, bit_resolution=12
):
	"""
	Analyze and visualize neural network fits across multiple tests with quantized weights and activations.
	This function fits a neural network model to sensor data and generates two graphs:
	1. ADC vs Instron Force (N) with the neural network fit overlaid.
	2. Residuals (ADC vs N minus neural fit) with optional smoothing.

	Parameters:
	- test_range: A range or list of test numbers to include in the analysis.
	- sensor_num: The specific sensor number to analyze.
	- units: Number of neurons in each hidden layer of the neural network.
	- layers: Number of hidden layers in the neural network model.
	- activation: Activation function for the neural network layers.
	- dropout_rate: Dropout rate for regularization.
	- l2_reg: L2 regularization factor to penalize large weights.
	- learning_rate: Learning rate for the optimizer.
	- epochs: Number of training epochs for the neural network.
	- batch_size: Batch size for training.
	- window_size: Window size for smoothing residuals.
	- poly_order: Polynomial order for Savitzky-Golay filter if smoothing method is 'savgol'.
	- smoothing_method: Method for smoothing residuals (e.g., 'savgol', 'boxcar').
	- save_graphs: Whether to save the graphs as a PDF.
	- show_graphs: Whether to display the graphs.
	- bit_resolution: Bit resolution for quantizing input and output data.
	"""
	
	# Initialize quantized neural network
	model = QuantizedNN(
		input_dim=1, units=units, layers=layers, activation=activation, dropout_rate=dropout_rate,
		weight_bit_width=bit_resolution, act_bit_width=bit_resolution
	)
	
	# Move model to device (CPU or GPU)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	
	# Define loss function and optimizer
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
	
	# Training loop over all tests
	for epoch in range(epochs):
		model.train()  # Set model to training mode
		total_loss = 0
		num_batches = 0
		
		for _TEST_NUM in test_range:
			# Load data
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			
			min_length = min(len(instron_data), len(updated_arduino_data))
			instron_force = instron_data["Force [N]"].iloc[:min_length].values
			updated_arduino_force = updated_arduino_data[f"ADC{sensor_num}"].iloc[:min_length].values
			
			# Quantize input and output data
			instron_force = quantize_data(instron_force, bit_resolution)
			updated_arduino_force = quantize_data(updated_arduino_force, bit_resolution)
			
			# Convert to PyTorch tensors
			instron_force = torch.Tensor(instron_force).unsqueeze(1).to(device)
			updated_arduino_force = torch.Tensor(updated_arduino_force).unsqueeze(1).to(device)
			
			# Create DataLoader
			dataset = torch.utils.data.TensorDataset(instron_force, updated_arduino_force)
			dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
			
			for x_batch, y_batch in dataloader:
				optimizer.zero_grad()
				outputs = model(x_batch)
				loss = criterion(outputs, y_batch)
				loss.backward()
				optimizer.step()
				
				total_loss += loss.item()
				num_batches += 1
		
		avg_loss = total_loss / num_batches if num_batches > 0 else 0
		if (epoch + 1) % 10 == 0:
			print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")
	
	# After training, evaluate and plot results
	model.eval()  # Set model to evaluation mode
	
	with PdfPages(f"/Users/jacobanderson/Downloads/Neural_Network_Fit_Sensor_Set_{sensor_num}.pdf") as pdf:
		plt.figure(figsize=(10, 6))
		
		# Plot ADC vs Instron Force (N) and overlay neural network fit
		for _TEST_NUM in test_range:
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			
			min_length = min(len(instron_data), len(updated_arduino_data))
			instron_force = instron_data["Force [N]"].iloc[:min_length].values
			updated_arduino_force = updated_arduino_data[f"ADC{sensor_num}"].iloc[:min_length].values
			
			instron_force_quantized = quantize_data(instron_force, bit_resolution)
			updated_arduino_force_quantized = quantize_data(updated_arduino_force, bit_resolution)
			
			instron_force_tensor = torch.Tensor(instron_force_quantized).unsqueeze(1).to(device)
			
			with torch.no_grad():
				outputs = model(instron_force_tensor).cpu().numpy()
			
			plt.plot(instron_force, updated_arduino_force, label=f"Test {_TEST_NUM} (ADC vs Instron)", linestyle='--', linewidth=1)
			plt.plot(instron_force, outputs, label=f"Neural Fit (Test {_TEST_NUM})", linewidth=2)
		
		plt.xlabel("Instron Force [N]")
		plt.ylabel("ADC Value")
		plt.legend(loc="upper left")
		plt.title(f"ADC vs Instron Force (Neural Fit) - Sensor {sensor_num}")
		plt.grid(True)
		
		if show_graphs:
			plt.show()
		
		if save_graphs:
			pdf.savefig()
		
		plt.close()
		
		# Plot residuals (ADC vs N minus neural fit)
		plt.figure(figsize=(10, 6))
		for _TEST_NUM in test_range:
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			
			min_length = min(len(instron_data), len(updated_arduino_data))
			instron_force = instron_data["Force [N]"].iloc[:min_length].values
			updated_arduino_force = updated_arduino_data[f"ADC{sensor_num}"].iloc[:min_length].values
			
			instron_force_quantized = quantize_data(instron_force, bit_resolution)
			updated_arduino_force_quantized = quantize_data(updated_arduino_force, bit_resolution)
			
			instron_force_tensor = torch.Tensor(instron_force_quantized).unsqueeze(1).to(device)
			
			with torch.no_grad():
				outputs = model(instron_force_tensor).cpu().numpy()
			
			residuals = updated_arduino_force_quantized - outputs.flatten()
			residuals_smoothed = apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
			
			plt.plot(instron_force, residuals_smoothed, label=f"Residuals (Test {_TEST_NUM})", linewidth=2)
		
		plt.xlabel("Instron Force [N]")
		plt.ylabel("Residuals (ADC - Neural Fit)")
		plt.legend(loc="upper left")
		plt.title(f"Residuals - Sensor {sensor_num}")
		plt.grid(True)
		
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
