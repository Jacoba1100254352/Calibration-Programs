import numpy as np
import matplotlib.pyplot as plt

# Data for bit resolutions and corresponding errors
bit_resolutions = np.array([8, 10, 12])  # Bit resolutions
mse_values = np.array([1.4e-5, 3e-6, 2e-6])  # Mean Squared Error (MSE) for each bit resolution
mae_values = np.array([0.0031, 0.0014, 0.0012])  # Mean Absolute Error (MAE) for each bit resolution

# Full-scale span (FSS) for SMT Load Cell (in Newtons)
full_scale_span_smt = 5.6 * 4.44822  # Convert 5.6 lbf to Newtons (5.6 lbf * 4.44822 N/lbf)
full_scale_span_fma = 5.0  # FMA Sensor's full-scale span in Newtons (5 N)

# Error percentages for the SMT Load Cell and FMA Sensor (without temperature effects)
nonlinearity_smt = 0.05 / 100  # ±0.05% FS
hysteresis_smt = 0.03 / 100  # ±0.03% FS
nonrepeatability_smt = 0.02 / 100  # ±0.02% FS

accuracy_fma = 2 / 100  # ±2% FSS for FMA sensor accuracy

smt_total_error_percentage = np.sqrt(nonlinearity_smt**2 + hysteresis_smt**2 + nonrepeatability_smt**2)

# Calculate force errors in Newtons (N)
nonlinearity_error_smt = nonlinearity_smt * full_scale_span_smt  # SMT nonlinearity in N
hysteresis_error_smt = hysteresis_smt * full_scale_span_smt  # SMT hysteresis in N
nonrepeatability_error_smt = nonrepeatability_smt * full_scale_span_smt  # SMT nonrepeatability in N

smt_total_error = np.sqrt(nonlinearity_error_smt**2 + hysteresis_error_smt**2 + nonrepeatability_error_smt**2)

# FMA sensor accuracy error in N (no TEB)
accuracy_error_fma = accuracy_fma * full_scale_span_fma  # FMA accuracy in N

# Combined worst-case error (RSS method)
combined_accuracy_