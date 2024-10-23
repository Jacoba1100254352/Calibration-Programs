import matplotlib.pyplot as plt
import numpy as np


# Set general plot appearance
SIZE_SMALL = 10
SIZE_DEFAULT = 14
SIZE_LARGE = 20
SIZE_XLARGE = 26
SIZE_XXLARGE = 32
SIZE_XXXLARGE = 40

plt.rc("font", family='Helvetica Neue', size=SIZE_DEFAULT, weight="bold")  # Default text sizes
plt.rc("axes", labelsize=SIZE_LARGE)  # X and Y labels fontsize
plt.rc("axes", linewidth=2.5)  # Line width for plot borders

# Data for bit resolutions and number of neurons
# neuron_counts = np.array(["Raw", 1, 2, 4, 8, 16, 32, 64, 128])  # Neuron counts, adjust as needed, bit: 12
# bit_resolutions = np.array(["Raw", 2, 4, 6, 8, 10, 12])  # Bit resolutions including "Uncalibrated", neuron: 160

# Originals
# rmse_bit_resolution = np.array([0.000761, 0.160526, 0.054192, 0.014177, 0.003746, 0.001705, 0.001739])  #relu: (with tanh hyperparams) RMSE for different bit resolutions, including uncalibrated
# rmse_neuron_count = np.array([0.000761, 0.097820, 0.065765, 0.053620, 0.010991, 0.012243, 0.006465, 0.004423, 0.005071])  # relu: (with new hyperparams (dedicated to test 9) 8-bit RMSE for different neuron counts  # activation='relu', l2_reg=0.0025, learning_rate=0.00025, epochs=100, mapping='N_vs_N', dropout_rate=0.15, layers=1, units=units, batch_size=16, bit_resolution=8

# Baseline based on original data at bit resolution
bit_resolutions = np.array(["B2", 2, "B4", 4, "B6", 6, "B8", 8, "B10", 10, "B12", 12])  # Bit resolutions including "Uncalibrated", neuron: 128
neuron_counts = np.array(["B", 1, 2, 4, 8, 16, 32, 64, 128])  # Neuron counts, adjust as needed, bit: 12
# rmse_neuron_count = np.array([0.003557, 0.003558, 0.003536, 0.003520, 0.003535, 0.003509, 0.003507, 0.003507, 0.003509])  # bit: 8
rmse_neuron_count = np.array([0.000822, 0.000822, 0.000665, 0.000561, 0.000449, 0.000438, 0.000434, 0.000433, 0.000433])  # bit: 12
rmse_bit_resolution = np.array([0.160485, 0.160576, 0.055334, 0.054070, 0.013829, 0.013781, 0.003557, 0.003509, 0.001175, 0.000947, 0.000822, 0.000433])  # neurons: 128


# New Setup (linear data)
# rmse_neuron_count = np.array([0.000761, 0.003558, 0.003536, 0.003520, 0.003535, 0.003509, 0.003507, 0.003507, 0.003509])  # bit: 8
# rmse_neuron_count = np.array([0.000761, 0.000822, 0.000665, 0.000561, 0.000449, 0.000438, 0.000434, 0.000433, 0.000433])  # bit: 12
# rmse_bit_resolution = np.array([0.000761, 0.160576, 0.054070, 0.013781, 0.003509, 0.000947, 0.000433])  # neurons: 128

# First 3: New Process
# Compare bits with baseline
# Compare neurons with 8 bits
# Compare neurons with 12 bits

# Next 2: Old process with unquantized baseline:
# Compare bits
# Compare neurons with 8 bits

# Last 3: New Process with unquantized baseline
# Compare bits with 128 neurons against unquantized baseline
# Compare neurons with 12 bits against unquantized baseline
# Compare neurons with 12 bits against unquantized baseline

# Last Last 6:

# New Setup (less linear data)
# neuron_counts = np.array(["Raw", 1, 2, 4, 8, 16, 32, 64, 128])  # Neuron counts, adjust as needed, bit: 12
# bit_resolutions = np.array(["Raw", 2, 4, 6, 8, 10, 12])  # Bit resolutions including "Uncalibrated", neuron: 160
# rmse_neuron_count = np.array([0.007314, 0.007987, 0.004266, 0.003995, 0.003842, 0.003977, 0.003752, 0.003775, 0.003757])  # bit: 8
# rmse_neuron_count = np.array([0.007314, 0.007311, 0.002580, 0.002082, 0.002073, 0.002036, 0.001378, 0.001477, 0.001449])  # bit: 12
# rmse_bit_resolution = np.array([0.007314, 0.170761, 0.055545, 0.013788, 0.003757, 0.001692, 0.001449])  # neurons: 128

# Less linear (To be filled in)
# neuron_counts = np.array(["B", 1, 2, 4, 8, 16, 32, 64, 128])  # Neuron counts, adjust as needed, bit: 8,12
# bit_resolutions = np.array(["B2", 2, "B4", 4, "B6", 6, "B8", 8, "B10", 10, "B12", 12])  # Bit resolutions including "Uncalibrated", neuron: 128
# rmse_neuron_count = np.array([0.007996, 0.007987, 0.004266, 0.003995, 0.003842, 0.003977, 0.003752, 0.003775, 0.003757])  # bit: 8
# rmse_neuron_count = np.array([0.007310, 0.007311, 0.002580, 0.002082, 0.002073, 0.002036, 0.001378, 0.001477, 0.001449])  # bit: 12
# rmse_bit_resolution = np.array([0.170667, 0.170761, 0.056869, 0.055545, 0.015462, 0.013788, 0.007996, 0.003757, 0.007362, 0.001692, 0.007310, 0.001449])  # neurons: 128


# Full-scale span (FSS) for FMA sensor (in Newtons)
full_scale_span_fma = 5.0  # FMA Sensor's full-scale span in Newtons (5 N)

# Error percentages for the FMA Sensor
accuracy_fma = 2 / 100  # ±2% FSS for FMA sensor accuracy
accuracy_error_fma = accuracy_fma * full_scale_span_fma  # FMA accuracy in N

# Wearable required accuracy based on 3 kPa with a 0.2 kPa tolerance
# For a platform area of 240 mm² (0.00024 m²)
wearable_required_accuracy = 0.048 / 4  # In Newtons (0.2 kPa tolerance * 0.00024 m²)

# Create the first plot: RMSE vs Bit Resolution (including "Uncalibrated")
residuals_fig, residuals_ax = plt.subplots(figsize=(10, 8))

# Plot RMSE vs Bit Resolution
residuals_ax.plot(bit_resolutions, rmse_bit_resolution, marker='o', color='black',
                  label='Calibrated Sensor', linewidth=2)

# Add horizontal lines for sensor accuracy and estimated wearable accuracy
# residuals_ax.axhline(y=accuracy_error_fma, color='r', linestyle='--',
#                      label=f'Sensor Datasheet Spec', linewidth=2)  # (±{accuracy_fma * 100:.0f}% FSS, ±{accuracy_error_fma:.2f} N)
residuals_ax.axhline(y=wearable_required_accuracy, color='b', linestyle='--',
                     label=f'Wearable Required Spec', linewidth=2)  # (±0.2 kPa, ±{wearable_required_accuracy:.4f} N)

# Set axis labels
# residuals_ax.set_xlabel('Bit Resolution', fontsize=12)
residuals_ax.set_ylabel(r'$\epsilon_{\text{rms}}$ (N)', fontsize=SIZE_XXXLARGE, labelpad=0)
residuals_ax.set_yscale('log')  # Log scale for better visibility
# residuals_ax.set_ylim(1e-4, 1e0)
residuals_ax.set_ylim(min(rmse_bit_resolution)*0.9, max(rmse_bit_resolution)*1.1)
residuals_ax.grid(True)

# Set bold and large font for tick labels
residuals_ax.tick_params(axis='both', which='major', labelsize=18, width=2.5, length=5, direction='in',
                         labelcolor='black', pad=10, top=True, bottom=True, left=True, right=True)
residuals_ax.tick_params(axis='both', which='minor', labelsize=14, width=1.5, length=2.5, direction='in',
                         labelcolor='black', top=True, bottom=True, left=True, right=True)

# Apply bold and Helvetica to tick labels using setp()
plt.setp(residuals_ax.get_xticklabels(), fontsize=SIZE_XXLARGE)
plt.setp(residuals_ax.get_yticklabels(), fontsize=SIZE_XXLARGE)

# xtick_labels = residuals_ax.get_xticklabels()
# for label in xtick_labels:
#     if label.get_text() == "Raw":
#         label.set_fontsize(SIZE_LARGE)  # Set to smaller size

# Set grid lines and add minor ticks
residuals_ax.grid(True, which='both', linestyle='-', linewidth=1.5)

# Formatter for scientific notation on Y axis
# residuals_ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# residuals_ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.yscale('log')  # Log scale for better visibility

# Add legend
# residuals_ax.legend()
residuals_ax.legend(
	# loc="upper right",
	# fontsize=SIZE_DEFAULT,
	prop={'family': 'Helvetica Neue', 'size': SIZE_XLARGE},  # Set font to Helvetica
	frameon=True,  # Enable the frame (box around the legend)
	edgecolor='black',  # Set the outline color
	framealpha=1,  # Set the transparency of the frame (1 = fully opaque)
	fancybox=False,  # Disable rounded corners
	shadow=False,  # No shadow
	facecolor='white',  # Background color of the legend box
	borderpad=0.5  # Padding inside the legend box
)

# Set the thickness of the legend box outline (bold)
legend = residuals_ax.get_legend()
legend.get_frame().set_linewidth(2.0)  # Increase the outline thickness

# Adjust layout
plt.tight_layout()

file_name = "rms_bit_relu"
# plt.savefig(f"/Users/jacobanderson/Documents/BYU Classes/Current BYU Classes/Research/Papers/{file_name}.pdf", dpi=300)

# Show plot
plt.show()

# Create the second plot: RMSE vs Number of Neurons
residuals_fig, residuals_ax = plt.subplots(figsize=(10, 8))

# Plot RMSE vs Neuron Count
residuals_ax.plot(neuron_counts, rmse_neuron_count, marker='o', color='black',
                  label='Calibrated Sensor', linewidth=2)

# Add horizontal lines for sensor accuracy and estimated wearable accuracy
# residuals_ax.axhline(y=accuracy_error_fma, color='r', linestyle='--',
#                      label=f'Sensor Datasheet Spec', linewidth=2)  # (±{accuracy_fma * 100:.0f}% FSS, ±{accuracy_error_fma:.2f} N)
residuals_ax.axhline(y=wearable_required_accuracy, color='b', linestyle='--',
                     label=f'Wearable Required Spec', linewidth=2)  # (±0.2 kPa, ±{wearable_required_accuracy:.4f} N)

# Set axis labels
# residuals_ax.set_xlabel('Number of Neurons', fontsize=12)
residuals_ax.set_ylabel(r'$\epsilon_{\text{rms}}$ (N)', fontsize=SIZE_XXXLARGE, labelpad=0)
residuals_ax.set_yscale('log')  # Log scale for better visibility
# residuals_ax.set_ylim(1e-3, 1e-2)
residuals_ax.set_ylim(min(rmse_neuron_count)*0.9, max(rmse_neuron_count)*1.1)
residuals_ax.grid(True)

# Set bold and large font for tick labels
residuals_ax.tick_params(axis='both', which='major', labelsize=18, width=2.5, length=5, direction='in',
                         labelcolor='black', pad=10, top=True, bottom=True, left=True, right=True)
residuals_ax.tick_params(axis='both', which='minor', labelsize=14, width=1.5, length=2.5, direction='in',
                         labelcolor='black', top=True, bottom=True, left=True, right=True)

# Apply bold and Helvetica to tick labels using setp()
plt.setp(residuals_ax.get_xticklabels(), fontsize=SIZE_XXLARGE)
plt.setp(residuals_ax.get_yticklabels(), fontsize=SIZE_XXLARGE)

# xtick_labels = residuals_ax.get_xticklabels()
# for label in xtick_labels:
#     if label.get_text() == "Raw":
#         label.set_fontsize(SIZE_LARGE)  # Set to smaller size
#         # label.set_position((label.get_position()[0] - 5, label.get_position()[1]))

# Set grid lines and add minor ticks
residuals_ax.grid(True, which='both', linestyle='-', linewidth=1.5)

# Formatter for scientific notation on Y axis
# residuals_ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# residuals_ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.yscale('log')  # Log scale for better visibility

# Add legend
# residuals_ax.legend()
residuals_ax.legend(
	# loc="upper right",
	# fontsize=SIZE_DEFAULT,
	prop={'family': 'Helvetica Neue', 'size': SIZE_XLARGE},  # Set font to Helvetica
	frameon=True,  # Enable the frame (box around the legend)
	edgecolor='black',  # Set the outline color
	framealpha=1,  # Set the transparency of the frame (1 = fully opaque)
	fancybox=False,  # Disable rounded corners
	shadow=False,  # No shadow
	facecolor='white',  # Background color of the legend box
	borderpad=0.5  # Padding inside the legend box
)

# Set the thickness of the legend box outline (bold)
legend = residuals_ax.get_legend()
legend.get_frame().set_linewidth(2.0)  # Increase the outline thickness

# Adjust layout
plt.tight_layout()

file_name = "rms_neuron_relu"
# plt.savefig(f"/Users/jacobanderson/Documents/BYU Classes/Current BYU Classes/Research/Papers/{file_name}.pdf", dpi=300)

# Show plot
plt.show()
