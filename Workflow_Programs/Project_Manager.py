from Sensor_Graphs import *
from Workflow_Programs.Align_Data import align_data
from Workflow_Programs.Apply_Calibrations_for_Graphing import apply_calibration_coefficients
from Workflow_Programs.Calculate_Linear_Fit_Sensor_Calibrations import calculate_coefficients
# from Workflow_Programs.Raw_to_CSV import write_raw_data_to_csv
from Workflow_Programs.Trim_Front_of_Data import trim_front_of_data
from Workflow_Programs.Truncate_Data import truncate_data


SAVE_GRAPHS = False

# Convert new data to CSV # Raw_to_CSV.py
# write_raw_data_to_csv()  # Original to Parsed

# Align data along the x (time) axis # Align_Data.py
align_data()  # Parsed to Aligned

# Trim the front of the data to a certain length or max force # Trim_Front_of_Data.py
trim_front_of_data(-0.001)  # Aligned to Aligned

# Truncate the data to a certain length or max force # Truncate_Data.py
truncate_data(-0.99)  # Aligned to Aligned

# Calculate new calibration coefficients # Calculate_Linear_Fit_Sensor_Calibrations.py
calculate_coefficients()  # N/A

# Apply new calibration coefficients to the data for graphing and verification # Apply_Calibrations_for_Graphing.py
apply_calibration_coefficients()  # Aligned to Calibrated

# STARTING_TEST = 9
# ENDING_TEST = 9
# TEST_RANGE = range(STARTING_TEST, ENDING_TEST + 1)

# Primary
# analyze_and_graph_residuals_and_fits_single_pdf_combined_multiple_tests(test_range=TEST_RANGE, save_graphs=SAVE_GRAPHS, smoothing_method=None)
analyze_and_graph_residuals_and_fits_individual_images(SAVE_GRAPHS)
# graph_sensor_data(SAVE_GRAPHS)
