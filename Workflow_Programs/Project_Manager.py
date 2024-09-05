from Sensor_Graphs import *


# 1766061.02
# 1766057 1766058 1766059

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

# Plot the data # Plot_Data.py
# graph_sensor_data(SAVE_GRAPHS)
# graph_and_verify_calibration()
# graph_sensor_data_difference()
# graph_sensor_average_best_fit()
# graph_sensor_average_error()
# graph_sensor_instron_error()
# graph_sensor_arduino_error()

STARTING_TEST = 9
ENDING_TEST = 13
TEST_RANGE = range(STARTING_TEST, ENDING_TEST + 1)

# Primary
# analyze_and_graph_residuals_and_fits_single_pdf_combined_multiple_tests(test_range=TEST_RANGE)

# To use the function with predefined parameters
# for _ in range(1):
# 	analyze_and_graph_neural_fit_single_pdf_combined_multiple_tests(
# 		test_range=TEST_RANGE, smoothing_method="boxcar", window_size=100, poly_order=1,
# 		sensor_num=2, layers=3, units=64, batch_size=16, l2_reg=0.001, epochs=200, learning_rate=0.0001, dropout_rate=0.2, activation="tanh", save_graphs=SAVE_GRAPHS,
# 	)

X_train, y_train = getTrainingData(_sensor_num=2, _TEST_RANGE=TEST_RANGE)
# To use the function with hyperparameter tuning
# analyze_and_graph_neural_fit_single_pdf_combined_multiple_tests(
# 	test_range=TEST_RANGE, smoothing_method="boxcar", window_size=100, poly_order=1,
# 	sensor_num=2, layers=4, units=128, batch_size=256, save_graphs=SAVE_GRAPHS,
# 	use_hyperparameter_tuning=False, X_train=X_train, y_train=y_train
# )

# analyze_and_graph_neural_fit_single_pdf_combined_multiple_tests(
#     test_range=TEST_RANGE,
#     smoothing_method="boxcar",
#     window_size=100,
#     poly_order=1,
#     sensor_num=2,
#     layers=4,  # Higher than best tuning results, consider if necessary
#     units=128,  # Consistent with the best results
#     activation='tanh',  # Consistent with the best results
#     dropout_rate=0.5,  # Higher due to more layers
#     l2_reg=0.0001,  # Middle ground based on previous results
#     learning_rate=0.0001,  # Consistent with the best results
#     epochs=100,  # Average of the best results
#     batch_size=256,  # Higher than best results, adjust if necessary
#     save_graphs=SAVE_GRAPHS,
#     use_hyperparameter_tuning=False,
#     X_train=X_train,
#     y_train=y_train
# )

analyze_and_graph_neural_fit_single_pdf_combined_multiple_tests(
    test_range=TEST_RANGE,
    smoothing_method="boxcar",
    window_size=100,
    poly_order=None,
    sensor_num=2,
    layers=1,  # Adjusted based on best scores
    units=128,  # Kept at 128 based on good performance
    batch_size=32,  # Reduced based on best parameters
    save_graphs=SAVE_GRAPHS,
    use_hyperparameter_tuning=False,  # Consider enabling tuning
    X_train=X_train,
    y_train=y_train,
    dropout_rate=0.2,  # Added based on tuning results
    l2_reg=0.0001,  # Added based on tuning results
    epochs=100,  # Specified based on typical good performance
    activation='tanh'  # Ensuring usage of 'tanh' as it performed well
)

# analyze_and_graph_neural_fit_per_test(
# 	test_range=TEST_RANGE, sensor_num=2, layers=2, units=128, activation='relu', dropout_rate=0.5,
# 	l2_reg=0.01, learning_rate=0.001, epochs=200, batch_size=32, window_size=100, poly_order=1,
# 	smoothing_method="boxcar", save_graphs=SAVE_GRAPHS, show_graphs=True, use_hyperparameter_tuning=False,
# 	X_train_data=X_train, y_train_data=y_train
# )

# train_and_graph_neural_fit_per_test(
#     test_range=TEST_RANGE, sensor_num=2, layers=6, units=164, activation='relu', learning_rate=0.001,
#     epochs=200, batch_size=32, save_graphs=SAVE_GRAPHS, show_graphs=True
# )

# analyze_and_graph_calibrated_data_and_fits_single_pdf_combined_multiple_tests(test_range=TEST_RANGE, smoothing_method='boxcar', window_size=700, poly_order=1, sensor_num=2, save_graphs=SAVE_GRAPHS)

# Others
# graph_sensor_average_error(test_range=TEST_RANGE)
# graph_sensor_data_difference(test_range=TEST_RANGE)
# analyze_and_graph_residuals_and_fits_single_pdf(test_range=TEST_RANGE)
# analyze_and_graph_residuals_and_fits_individual_images(SAVE_GRAPHS)
# analyze_and_graph_residuals_and_fits_individual_images(SAVE_GRAPHS, False)

# plot_adjusted_linear_fits_no_offsets(test_range=TEST_RANGE, save_graphs=SAVE_GRAPHS, useArduinoADC=False)
# plot_adjusted_linear_fits(test_range=TEST_RANGE)

# Simple
# plot_Instron_force_vs_Arduino_force(test_num=10, sensor_num=2, show_polyfit=True, order=1)

# After applying calibrations from previous test, N will now be using that, graph N against the Instron to verify the calibration
