import time

import smbus2


# Assuming bus 1 for Raspberry Pi models 2, 3, 4, and 5
bus = smbus2.SMBus(1)

# DS3231 I2C address
address = 0x68


def bcd_to_dec(val):
	return (val // 16 * 10) + (val % 16)


def get_rtc_time():
	# Read RTC time registers
	rtc_time = bus.read_i2c_block_data(address, 0, 7)
	
	# Convert BCD to decimal
	seconds = bcd_to_dec(rtc_time[0] & 0x7F)
	minutes = bcd_to_dec(rtc_time[1])
	hours = bcd_to_dec(rtc_time[2] & 0x3F)
	day = bcd_to_dec(rtc_time[4])
	month = bcd_to_dec(rtc_time[5])
	year = bcd_to_dec(rtc_time[6]) + 2000  # Assuming RTC stores the year as offset from 2000
	
	# Create a full time tuple including ignored values
	time_tuple = (year, month, day, hours, minutes, seconds, -1, -1, -1)
	return time.mktime(time_tuple)  # Converts time_tuple to epoch time


def get_system_time_ms():
	# Get current system time in milliseconds
	return int(time.time() * 1000)


def wait_for_next_second_flip():
	# Wait until the next second flip
	previous_second = time.localtime().tm_sec
	while True:
		current_second = time.localtime().tm_sec
		if current_second != previous_second:
			break


def measure_ms_difference():
	# Record the RTC and system times at roughly the same moment
	rtc_time_at_check = get_rtc_time()
	system_time_at_check = time.time()
	
	# Wait for the next second flip on both RTC and system clock
	wait_for_next_second_flip()
	rtc_time_before = get_rtc_time()
	wait_for_next_second_flip()
	rtc_time_after = get_rtc_time()
	system_time_before = get_system_time_ms()
	wait_for_next_second_flip()
	system_time_after = get_system_time_ms()
	
	# Calculate the differences
	rtc_ms_difference = rtc_time_after - rtc_time_before
	system_ms_difference = system_time_after - system_time_before
	time_difference_seconds = system_time_at_check - rtc_time_at_check
	
	print("RTC Time:", time.strftime("%H:%M:%S %d/%m/%Y", time.gmtime(rtc_time_after)))
	print("System Time:", time.strftime("%H:%M:%S %d/%m/%Y", time.localtime()))
	print("Millisecond Difference (RTC - System):", rtc_ms_difference - system_ms_difference)
	print("Time Difference in Seconds (System - RTC):", time_difference_seconds)


# Call the function to measure the milliseconds difference
measure_ms_difference()
print(f"RTC Time s: {get_rtc_time()}")
