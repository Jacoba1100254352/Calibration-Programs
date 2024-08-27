import smbus2
import time

# Replace 1 with 0 if using an older Raspberry Pi model
bus = smbus2.SMBus(1)

# DS3231 I2C address
address = 0x68

# Function to convert decimal to binary coded decimal (BCD)
def dec_to_bcd(val):
    return (val // 10 * 16) + (val % 10)

# Function to set the time on the RTC
def set_rtc_time():
    # Get the current time
    t = time.gmtime()  # Use gmtime or localtime depending on your RTC's time zone setting
    # Write time to RTC (hour, minute, second)
    bus.write_byte_data(address, 0x00, dec_to_bcd(t.tm_sec))
    bus.write_byte_data(address, 0x01, dec_to_bcd(t.tm_min))
    bus.write_byte_data(address, 0x02, dec_to_bcd(t.tm_hour))
    # Optionally set the date as well
    bus.write_byte_data(address, 0x04, dec_to_bcd(t.tm_mday))
    bus.write_byte_data(address, 0x05, dec_to_bcd(t.tm_mon))
    bus.write_byte_data(address, 0x06, dec_to_bcd(t.tm_year - 2000))
    print("RTC module set successfully to: " + str(t))

# Set the RTC time
set_rtc_time()
