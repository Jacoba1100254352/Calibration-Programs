import random

from Sensor_Graphs import *


SAVE_GRAPHS = False

# Convert new data to CSV # Raw_to_CSV.py
# write_raw_data_to_csv()  # Original to Parsed

# # Align data along the x (time) axis # Align_Data.py
# align_data()  # Parsed to Aligned
#
# # Trim the front of the data to a certain length or max force # Trim_Front_of_Data.py
# trim_front_of_data(-0.001)  # Aligned to Aligned
#
# # Truncate the data to a certain length or max force # Truncate_Data.py
# truncate_data(-0.99)  # Aligned to Aligned
#
# # Calculate new calibration coefficients # Calculate_Linear_Fit_Sensor_Calibrations.py
# calculate_coefficients()  # N/A
#
# # Apply new calibration coefficients to the data for graphing and verification # Apply_Calibrations_for_Graphing.py
# apply_calibration_coefficients()  # Aligned to Calibrated

STARTING_TEST = 9
ENDING_TEST = 13
TEST_RANGE = range(STARTING_TEST, ENDING_TEST + 1)

# Primary
# analyze_and_graph_residuals_and_fits_single_pdf_combined_multiple_tests(test_range=TEST_RANGE)
# analyze_and_graph_residuals_and_fits_individual_images(SAVE_GRAPHS, False)

torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

if torch.cuda.is_available():
	torch.cuda.manual_seed(seed_value)
	torch.cuda.manual_seed_all(seed_value)

# Secondary Analyses
analyze_and_graph_neural_fit(
	test_range=TEST_RANGE, sensor_num=2, save_graphs=SAVE_GRAPHS,
	smoothing_method="boxcar", activation='tanh',
	l2_reg=0.005, learning_rate=0.00075, epochs=100, mapping='N_vs_N',
	dropout_rate=0.1, layers=1, units=160, batch_size=64, bit_resolution=8, enable_hyperparameter_tuning=False
)

# analyze_and_graph_calibrated_data_and_fits_single_pdf_combined_multiple_tests(
# 	test_range=TEST_RANGE, sensor_num=2, save_graphs=SAVE_GRAPHS,
# 	smoothing_method='boxcar', bit_resolution=12, mapping='N_vs_N'
# )
