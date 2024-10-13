import matplotlib.pyplot as plt
import numpy as np


# Test 9
# Data for bit resolutions and number of neurons
bit_resolutions = np.array(["Uncalibrated", 2, 4, 6, 8, 12])  # Bit resolutions including "Uncalibrated", neuron: 160
neuron_counts = np.array(["Uncalibrated", 1, 2, 4, 6, 8, 16, 32, 64, 128])  # Neuron counts, adjust as needed, bit: 12

# Example RMSE data (replace these with your actual computed RMSE values)
rmse_bit_resolution = np.array([1.184221, 0.160526, 0.054192, 0.014177, 0.003746,
                                0.001739])  # RMSE for different bit resolutions, including uncalibrated
rmse_neuron_count = np.array([1.184221, 0.035187, 0.021183, 0.013987, 0.010988, 0.009300, 0.006065, 0.002957, 0.003753,
                              0.002102])  # RMSE for different neuron counts

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

# SMT total error using RSS method
smt_total_error = np.sqrt(nonlinearity_error_smt**2 + hysteresis_error_smt**2 + nonrepeatability_error_smt**2)

# FMA sensor accuracy error in N (no TEB)
accuracy_error_fma = accuracy_fma * full_scale_span_fma  # FMA accuracy in N

# Combined worst-case error (RSS method)
combined_accuracy_error = np.sqrt(accuracy_error_fma**2 + smt_total_error**2)

# Create the first plot: RMSE vs Bit Resolution (including "Uncalibrated")
plt.figure(figsize=(10, 6))
plt.plot(bit_resolutions, rmse_bit_resolution, marker='o', color='black', label='RMSE vs Bit Resolution', linewidth=2)

# Add horizontal lines for each error type
plt.axhline(y=smt_total_error, color='g', linestyle='--', label=f'SMT Load Cell Total Error (±{smt_total_error_percentage:.4f}% FS, ±{smt_total_error:.2f} N)')
plt.axhline(y=accuracy_error_fma, color='m', linestyle='--', label=f'FMA Sensor Accuracy (±{accuracy_fma * 100:.0f}% FSS, ±{accuracy_error_fma:.2f} N)')
plt.axhline(y=combined_accuracy_error, color='orange', linestyle='--', label=f'Combined SMT + FMA Accuracy Error (±{combined_accuracy_error:.3f} N)')

# Add labels and title
# plt.title('RMS Error Across Different Bit Resolutions Compared to Sensor Specifications (Force Units)', fontsize=14)
plt.xlabel('Bit Resolution', fontsize=12)
plt.ylabel('RMSE (Force in N)', fontsize=12)
plt.yscale('log')  # Log scale for better visibility
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Create the second plot: RMSE vs Number of Neurons
plt.figure(figsize=(10, 6))
plt.plot(neuron_counts, rmse_neuron_count, marker='o', color='black', label='RMSE vs Neuron Count', linewidth=2)

# Add horizontal lines for each error type
plt.axhline(y=smt_total_error, color='g', linestyle='--', label=f'SMT Load Cell Total Error (±{smt_total_error_percentage:.4f}% FS, ±{smt_total_error:.2f} N)')
plt.axhline(y=accuracy_error_fma, color='m', linestyle='--', label=f'FMA Sensor Accuracy (±{accuracy_fma * 100:.0f}% FSS, ±{accuracy_error_fma:.2f} N)')
plt.axhline(y=combined_accuracy_error, color='orange', linestyle='--', label=f'Combined SMT + FMA Accuracy Error (±{combined_accuracy_error:.3f} N)')

# Add labels and title
# plt.title('RMSE Across Different Number of Neurons', fontsize=14)
plt.xlabel('Number of Neurons', fontsize=12)
plt.ylabel('RMSE (Force in N)', fontsize=12)
plt.yscale('log')  # Log scale for better visibility
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
