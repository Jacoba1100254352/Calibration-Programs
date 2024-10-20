import random

from Sensor_Graphs import *
from Workflow_Programs.Supplemental_Sensor_Graphs import analyze_and_graph_residuals_and_fits_single_pdf_combined_multiple_tests


SAVE_GRAPHS = False

# # Convert new data to CSV # Raw_to_CSV.py
# # write_raw_data_to_csv()  # Original to Parsed
#
# Align data along the x (time) axis # Align_Data.py
# align_data()  # Parsed to Aligned
#
# # Trim the front of the data to a certain length or max force # Trim_Front_of_Data.py
# trim_front_of_data(-0.001)  # Aligned to Aligned
#
# # Truncate the data to a certain length or max force # Truncate_Data.py
# truncate_data(-0.99)  # Aligned to Aligned
#
# Calculate new calibration coefficients # Calculate_Linear_Fit_Sensor_Calibrations.py
# calculate_coefficients()  # N/A
#
# # Apply new calibration coefficients to the data for graphing and verification # Apply_Calibrations_for_Graphing.py
# apply_calibration_coefficients()  # Aligned to Calibrated

STARTING_TEST = 9
ENDING_TEST = 9
TEST_RANGE = range(STARTING_TEST, ENDING_TEST + 1)

# # Primary
# analyze_and_graph_residuals_and_fits_single_pdf_combined_multiple_tests(test_range=TEST_RANGE, save_graphs=SAVE_GRAPHS, smoothing_method=None)
# analyze_and_graph_residuals_and_fits_individual_images(SAVE_GRAPHS, False)  # 0.39 to 0.79
# graph_sensor_data(SAVE_GRAPHS)
# RMSE=1.370042
# RMSE=0.418916

# Sensor 1 looks more or less ok, but should probably have 0.4 to 0.8 removed
# Sensor 2 looks fine
# Sensor 3 definitely needs 0.4 to 0.8 removed
# Sensor 4 should have 0.39 to 0.79 removed

# Secondary Analyses
analyze_and_graph_neural_fit(
	test_range=TEST_RANGE, sensor_num=2, save_graphs=SAVE_GRAPHS,
	activation='relu', l2_reg=0.0025, learning_rate=0.00025, epochs=100, mapping='N_vs_N',
	dropout_rate=0.15, layers=1, units=64, batch_size=16, bit_resolution=8, save_bit=True
)

# for bit in [6, 8, 12]:
# 	analyze_and_graph_neural_fit(
# 		test_range=TEST_RANGE, sensor_num=2, save_graphs=SAVE_GRAPHS,  # enable_hyperparameter_tuning=True,
# 		activation='relu', l2_reg=0.005, learning_rate=0.0005, epochs=100, mapping='N_vs_N',
# 		dropout_rate=0.1, layers=1, units=64, batch_size=32, bit_resolution=bit, save_bit=True
# 	)
#
# for units in [8, 32, 256]:
# 	analyze_and_graph_neural_fit(
# 		test_range=TEST_RANGE, sensor_num=2, save_graphs=SAVE_GRAPHS,  # enable_hyperparameter_tuning=True,
# 		activation='relu', l2_reg=0.005, learning_rate=0.0005, epochs=100, mapping='N_vs_N',
# 		dropout_rate=0.1, layers=1, units=units, batch_size=32, bit_resolution=12, save_bit=True
# 	)

# for units in [1, 2, 4, 8, 16, 32, 64, 128]:
# 	analyze_and_graph_neural_fit(
# 		test_range=TEST_RANGE, sensor_num=2, save_graphs=SAVE_GRAPHS,
# 		activation='relu', l2_reg=0.0025, learning_rate=0.00025, epochs=100, mapping='N_vs_N',
# 		dropout_rate=0.15, layers=1, units=units, batch_size=16, bit_resolution=8, save_bit=True
# 	)
