from keras.api.callbacks import EarlyStopping
from keras.api.layers import Dense
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler

from Neural_Net_Code.Neural_Fit_Supporting_Functions import build_neural_network, hyperparameter_tuning
from Workflow_Programs.Configuration_Variables import *
from Workflow_Programs.Supporting_and_Archived_Graphing_Functions.Supplemental_Sensor_Graph_Functions import apply_smoothing


def analyze_and_graph_neural_fit_per_test(
	test_range, sensor_num, layers=2, units=64, activation='relu', dropout_rate=0.5, l2_reg=0.01,
	learning_rate=0.001, epochs=100, batch_size=32, window_size=None, poly_order=None,
	smoothing_method="boxcar", save_graphs=True, show_graphs=True, use_hyperparameter_tuning=False,
	X_train_data=None, y_train_data=None
):
	"""
	Analyze and visualize residuals and neural network fits for each test by training a separate
	neural network for each test, with optional hyperparameter tuning for each test.

	Parameters:
	- test_range: A range or list of test numbers to include in the analysis.
	- sensor_num: The sensor number to analyze.
	- layers: Number of hidden layers in the neural network. (Recommended: 1-3)
	- units: Number of units in each hidden layer. (Recommended: 32-128)
	- activation: Activation function for hidden layers. (Recommended: 'relu', 'tanh', or 'sigmoid')
	- dropout_rate: Dropout rate for regularization. (Recommended: 0.2-0.7)
	- l2_reg: L2 regularization parameter. (Recommended: 0.0001-0.01)
	- learning_rate: Learning rate for the optimizer. (Recommended: 0.0001-0.01)
	- epochs: Number of training epochs. (Recommended: 50-200)
	- batch_size: Size of the training batch. (Recommended: 16-64)
	- window_size: Window size for the smoothing operation. If None, no smoothing is applied.
	- poly_order: Polynomial order for the Savitzky-Golay filter. (Typical: 2-3)
	- smoothing_method: The smoothing method ('savgol', 'boxcar', 'median', or None).
	- save_graphs: Boolean flag to save the graphs as a PDF.
	- show_graphs: Boolean flag to display the graphs during the process.
	- use_hyperparameter_tuning: Boolean flag to indicate whether to use hyperparameter tuning.
	- X_train_data, y_train_data: Training data for hyperparameter tuning for each test.

	Returns:
	- None. Generates graphs for each test showing the residuals for the fitted neural network.
	"""
	
	scaler = StandardScaler()
	
	with PdfPages(f"/Users/jacobanderson/Downloads/Neural_Network_Fit_Sensor_{sensor_num}_Per_Test.pdf") as pdf:
		for _TEST_NUM in test_range:
			# Load data for the current test
			instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
			
			# Ensure arrays are of equal length for accurate comparison
			min_length = min(len(instron_data), len(updated_arduino_data))
			instron_force = instron_data["Force [N]"].iloc[:min_length].values.reshape(-1, 1)
			updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"].iloc[
			                        :min_length].values.reshape(-1, 1)
			
			# Standardize the input data for the current test
			instron_force_scaled = scaler.fit_transform(instron_force)
			
			# Optional hyperparameter tuning for each test
			if use_hyperparameter_tuning and X_train_data is not None and y_train_data is not None:
				best_model = hyperparameter_tuning(X_train_data, y_train_data, input_dim=1)
				model = best_model.model
			else:
				# Build and train the neural network for each test using the passed values
				model = build_neural_network(input_dim=1, layers=layers, units=units, activation=activation,
				                             dropout_rate=dropout_rate, l2_reg=l2_reg, learning_rate=learning_rate)
				early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
				model.fit(instron_force_scaled, updated_arduino_force, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[
					early_stopping])
			
			# Predict the fitted values for the current test
			fitted_values = model.predict(instron_force_scaled)
			residuals = updated_arduino_force - fitted_values
			
			# Apply smoothing to residuals, if necessary
			residuals_smoothed = apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
			
			# Ensure instron_force and residuals_smoothed have the same length
			instron_force_plot = instron_force[:len(residuals_smoothed)]
			
			# Plot residuals for the current test on the same graph
			plt.plot(instron_force_plot, residuals_smoothed, label=f"Test {_TEST_NUM}", linewidth=2)
		
		# Finalize the plot
		plt.xlabel("Calibration Force [N]")
		plt.ylabel("Residual Force [N]")
		plt.title(f"Residuals for Neural Fit for Sensor {sensor_num} Across All Tests")
		plt.legend(loc="lower left")
		plt.grid(True)
		plt.gca().invert_xaxis()
		
		# Show or save graphs as specified
		if show_graphs:
			plt.show()
		if save_graphs:
			pdf.savefig()
		plt.close()


def train_and_graph_neural_fit_per_test(
	test_range, sensor_num, layers=6, units=164, activation='relu', learning_rate=0.001, epochs=100, batch_size=32,
	smoothing_method="boxcar",
	window_size=100, poly_order=1, save_graphs=True, show_graphs=True, X_train_data=None, y_train_data=None
):
	"""
	Train a neural network for each test and graph all predicted fits together on the same plot.

	Parameters:
	- test_range: A range or list of test numbers to include in the analysis.
	- sensor_num: The sensor number to analyze.
	- layers: Number of hidden layers in the neural network. (Fixed at 6 here for this guide)
	- units: Number of units in each hidden layer. (Fixed at 164 here for this guide)
	- activation: Activation function for hidden layers. ('relu' used here)
	- learning_rate: Learning rate for the optimizer. (Recommended: 0.001)
	- epochs: Number of training epochs. (Recommended: 50-200)
	- batch_size: Size of the training batch. (Recommended: 16-64)
	- save_graphs: Boolean flag to save the graph as a PDF.
	- show_graphs: Boolean flag to display the graph during the process.
	- X_train_data, y_train_data: Optionally passed training data for each test.

	Returns:
	- None. Generates a graph showing all fits on the same plot.
	"""
	
	scaler = StandardScaler()
	
	# Create figure for plotting all fits together
	plt.figure(figsize=(15, 8))
	
	for _TEST_NUM in test_range:
		# Load data for the current test
		instron_data = pd.read_csv(get_data_filepath(ALIGNED_INSTRON_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
		updated_arduino_data = pd.read_csv(get_data_filepath(CALIBRATED_ARDUINO_DIR, sensor_num, _TEST_NUM=_TEST_NUM))
		
		# Ensure arrays are of equal length for accurate comparison
		min_length = min(len(instron_data), len(updated_arduino_data))
		instron_force = instron_data["Force [N]"].iloc[:min_length].values.reshape(-1, 1)
		updated_arduino_force = updated_arduino_data["Force [N]" if SIMPLIFY else f"Force{sensor_num} [N]"].iloc[
		                        :min_length].values.reshape(-1, 1)
		
		# Standardize the input data for the current test
		instron_force_scaled = scaler.fit_transform(instron_force)
		
		# Build a new model for each test
		model = Sequential()
		model.add(Dense(units=1, input_shape=([1]), activation='linear'))  # Input layer
		for _ in range(layers):
			model.add(Dense(units=units, activation=activation))  # Hidden layers
		model.add(Dense(1))  # Output layer for regression
		
		# Compile the model
		model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
		
		# Train the model on this test's data
		model.fit(instron_force_scaled, updated_arduino_force, epochs=epochs, batch_size=batch_size, verbose=0)
		
		# Predict the fitted values for the current test
		fitted_values = model.predict(instron_force_scaled)
		residuals = updated_arduino_force - fitted_values
		
		# Apply smoothing to residuals, if necessary
		residuals_smoothed = residuals  # apply_smoothing(residuals, method=smoothing_method, window_size=window_size, poly_order=poly_order)
		
		# Ensure instron_force and residuals_smoothed have the same length
		instron_force_plot = instron_force[:len(residuals_smoothed)]
		
		# Plot the results for this test
		plt.plot(instron_force_plot, residuals_smoothed, label=f"Test {_TEST_NUM}", linewidth=2)
	
	# Finalize the graph
	plt.xlabel("Calibration Force [N]")
	plt.ylabel("Predicted Force [N]")
	plt.title(f"Neural Network Fits for Sensor {sensor_num} Across Tests")
	plt.legend(loc="lower right")
	plt.grid(True)
	
	# Show or save graph as specified
	if show_graphs:
		plt.show()
	if save_graphs:
		plt.savefig(f"/Users/jacobanderson/Downloads/Neural_Fits_Sensor_{sensor_num}_All_Tests.pdf")
	plt.close()
	
	print("Completed training and graphing fits for each test.")
