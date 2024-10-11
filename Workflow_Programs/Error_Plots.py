import matplotlib.pyplot as plt
import numpy as np


# Data for bit resolutions and corresponding errors
bit_resolutions = np.array(["Uncalibrated", 8, 12])  # Bit resolutions
mse_values = np.array([1.402379, 1.4e-5, 2e-6])  # Mean Squared Error (MSE) for each bit resolution
mae_values = np.array([1.040731, 0.0031, 0.0012])  # Mean Absolute Error (MAE) for each bit resolution

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
combined_accuracy_error = np.sqrt(accuracy_error_fma**2 + smt_total_error**2)

# Print calculated errors
print(f"SMT Load Cell Total Error: {smt_total_error:.6f} N")
print(f"FMA Sensor Accuracy Error: {accuracy_error_fma:.6f} N")
print(f"Combined SMT + FMA Accuracy Error: {combined_accuracy_error:.6f} N")

# Create a plot for MSE and MAE
plt.figure(figsize=(10, 6))
plt.plot(bit_resolutions, mse_values, marker='o', color='b', label='MSE', linewidth=2)
plt.plot(bit_resolutions, mae_values, marker='s', color='r', label='MAE', linewidth=2)

# Add horizontal lines for each error type
plt.axhline(y=smt_total_error, color='g', linestyle='--', label=f'SMT Load Cell Total Error (±{smt_total_error_percentage:.4f}% FS, ±{smt_total_error:.2f} N)')
plt.axhline(y=accuracy_error_fma, color='m', linestyle='--', label=f'FMA Sensor Accuracy (±{accuracy_fma * 100:.0f}% FSS, ±{accuracy_error_fma:.2f} N)')
plt.axhline(y=combined_accuracy_error, color='orange', linestyle='--', label=f'Combined SMT + FMA Accuracy Error (±{combined_accuracy_error:.3f} N)')

# Add labels and title
plt.title('Errors Across Different Bit Resolutions Compared to Sensor Specifications (Force Units)', fontsize=14)
plt.xlabel('Bit Resolution', fontsize=12)
plt.ylabel('Error (Force in N)', fontsize=12)
plt.yscale('log')  # Use log scale for better visibility
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
