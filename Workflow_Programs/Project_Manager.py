from Sensor_Graphs import *


# 1766061.02
# 1766057 1766058 1766059

SAVE_GRAPHS = True

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
# analyze_and_graph_calibrated_data_and_fits_single_pdf_combined_multiple_tests(test_range=TEST_RANGE, window_size=100, poly_order=1, sensor_num=2, save_graphs=SAVE_GRAPHS)

# Others
# analyze_and_graph_residuals_and_fits_single_pdf(test_range=TEST_RANGE)
# analyze_and_graph_residuals_and_fits_individual_images(SAVE_GRAPHS)
# analyze_and_graph_residuals_and_fits_individual_images(SAVE_GRAPHS, False)

plot_adjusted_linear_fits_no_offsets(test_range=TEST_RANGE, save_graphs=SAVE_GRAPHS, useArduinoADC=False)
# plot_adjusted_linear_fits(test_range=TEST_RANGE)

# Simple
# plot_Instron_force_vs_Arduino_force(test_num=10, sensor_num=2, show_polyfit=True, order=1)

# After applying calibrations from previous test, N will now be using that, graph N against the Instron to verify the calibration
