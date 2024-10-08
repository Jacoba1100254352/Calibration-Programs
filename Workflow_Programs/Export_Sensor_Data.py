import os

import serial

from Configuration_Variables import get_data_filepath, ORIGINAL_ARDUINO_DIR


# Test 12 Specimen/sensor test order: 1, 4, 3, 2
# Sensor Set 2, Test 1-4 on Sensor 4 are done without zeroing, tho as of test 2 no drift noticed (or very minor positive)

# Value of drift at the end of the first test was 0.0011
# Value of drift at the end of the first test was 0.0008
# Value of drift at the end of the second test was 0.0013


# Modify the port and baud rate as per your setup
PORT = "/dev/cu.usbserial-10"  # "/dev/cu.usbserial-1420"  # On Windows, this is usually "COMX" where X is a number. # On Mac/Linux, it might be "/dev/ttyUSB0" or similar.
BAUD_RATE = 57600
SENSOR_NUM = 4
OUTPUT_FILE = get_data_filepath(ORIGINAL_ARDUINO_DIR, SENSOR_NUM)


def main():
	# Open the serial port
	with serial.Serial(PORT, BAUD_RATE, timeout=1) as ser:
		print(f"Connected to {PORT} at {BAUD_RATE} baud rate.")
		
		# Ensure the directory exists
		directory = os.path.dirname(OUTPUT_FILE)
		if not os.path.exists(directory):
			os.makedirs(directory)
		
		# Wait for the user to press the 'r' key
		while True:
			key = input("Press 'r' to start reading data...")
			if key.lower() == 'r':
				break
		
		# Open the output file in append mode
		with open(OUTPUT_FILE, 'a') as file:
			try:
				while True:
					# Read a line from the serial port
					line = ser.readline().decode("utf-8").strip()
					
					# If the line is not empty, print and save it
					if line:
						print(line)
						file.write(line + "\n")
			
			except KeyboardInterrupt:
				print("Exiting...")


if __name__ == "__main__":
	main()
