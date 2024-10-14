import matplotlib.pyplot as plt
import numpy as np


# Data for bit resolutions and number of neurons
bit_resolutions = np.array(["Uncalibrated", 2, 4, 6, 8, 10, 12])  # Bit resolutions including "Uncalibrated", neuron: 160
neuron_counts = np.array(["Uncalibrated", 1, 2, 4, 6, 8, 16, 32, 64, 128])  # Neuron counts, adjust as needed, bit: 12

# Example RMSE data (replace these with your actual computed RMSE values)
rmse_bit_resolution = np.array([1.184221, 0.160526, 0.054192, 0.014177, 0.003746, 0.001705,
                                0.001739])  # tanh: RMSE for different bit resolutions, including uncalibrated
rmse_neuron_count = np.array([1.184221, 0.035187, 0.021183, 0.013987, 0.010988, 0.009300, 0.006065, 0.002957, 0.003753,
                              0.002102])  # tanh: RMSE for different neuron counts

# rmse_bit_resolution = np.array([1.184221, 0.162081, 0.054298, 0.014876, 0.005299, 0.005331, 0.005084])  # relu: RMSE for different bit resolutions, including uncalibrated
# rmse_neuron_count = np.array([1.184221, 0.280838, 0.280838, 0.075722, 0.013051, 0.015576, 0.005593, 0.004132, 0.005247, 0.003716])  # relu: RMSE for different neuron counts

# Full-scale span (FSS) for FMA sensor (in Newtons)
full_scale_span_fma = 5.0  # FMA Sensor's full-scale span in Newtons (5 N)

# Error percentages for the FMA Sensor
accuracy_fma = 2 / 100  # ±2% FSS for FMA sensor accuracy
accuracy_error_fma = accuracy_fma * full_scale_span_fma  # FMA accuracy in N

# Wearable required accuracy based on 3 kPa with a 0.2 kPa tolerance
# For a platform area of 240 mm² (0.00024 m²)
wearable_required_accuracy = 0.048  # In Newtons (0.2 kPa tolerance * 0.00024 m²)

# Create the first plot: RMSE vs Bit Resolution (including "Uncalibrated")
plt.figure(figsize=(10, 6))
plt.plot(bit_resolutions, rmse_bit_resolution, marker='o', color='black', label=r'$\epsilon_{\text{rms}}$ vs Bit Resolution', linewidth=2)

# Add horizontal lines for sensor accuracy and estimated wearable accuracy
plt.axhline(y=accuracy_error_fma, color='m', linestyle='--', label=f'FMA Sensor Accuracy (±{accuracy_fma * 100:.0f}% FSS, ±{accuracy_error_fma:.2f} N)')
plt.axhline(y=wearable_required_accuracy, color='b', linestyle='--', label=f'Wearable Required Accuracy (±0.2 kPa, ±{wearable_required_accuracy:.4f} N)')

# Add labels and title
# plt.title('RMS Error Across Different Bit Resolutions Compared to Sensor Specifications (Force Units)', fontsize=14)
plt.xlabel('Bit Resolution', fontsize=12)
plt.ylabel(r'$\epsilon_{\text{rms}}$ (Force in N)', fontsize=12)
plt.yscale('log')  # Log scale for better visibility
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Create the second plot: RMSE vs Number of Neurons
plt.figure(figsize=(10, 6))
plt.plot(neuron_counts, rmse_neuron_count, marker='o', color='black', label=r'$\epsilon_{\text{rms}}$ vs Neuron Count', linewidth=2)

# Add horizontal lines for sensor accuracy and estimated wearable accuracy
plt.axhline(y=accuracy_error_fma, color='m', linestyle='--', label=f'FMA Sensor Accuracy (±{accuracy_fma * 100:.0f}% FSS, ±{accuracy_error_fma:.2f} N)')
plt.axhline(y=wearable_required_accuracy, color='b', linestyle='--', label=f'Wearable Required Accuracy (±0.2 kPa, ±{wearable_required_accuracy:.4f} N)')

# Add labels and title
# plt.title('RMSE Across Different Number of Neurons', fontsize=14)
plt.xlabel('Number of Neurons', fontsize=12)
plt.ylabel(r'$\epsilon_{\text{rms}}$ (Force in N)', fontsize=12)
plt.yscale('log')  # Log scale for better visibility
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
