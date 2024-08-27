import smbus2
import time

# Assuming bus 1 for Raspberry Pi models 2, 3, 4, and 5
bus = smbus2.SMBus(1)

# DS3231 I2C address
address = 0x68

def dec_to_bcd(val):
    return (val // 10 * 16) + (val % 10)

def set_rtc_time():
    # Wait until we're just past the start of a new second (e.g., 0.1 seconds past)
    while time.time() % 1 > 0.1:
        # Busy wait
        pass
    
    # The moment we exit the loop, we're right after the start of a new second
    # Sleep a very short amount to ensure we are not setting the time in the middle of a second
    time.sleep(0.02)  # 50 milliseconds to ensure we're within the first part of the new second but past the transition

    # Get the current time, now accurately at the start of a second
    precise_time = time.gmtime()  # Use gmtime or localtime as needed
    
    # Write the time to the RTC
    bus.write_byte_data(address, 0x00, dec_to_bcd(precise_time.tm_sec))
    bus.write_byte_data(address, 0x01, dec_to_bcd(precise_time.tm_min))
    bus.write_byte_data(address, 0x02, dec_to_bcd(precise_time.tm_hour))
    bus.write_byte_data(address, 0x04, dec_to_bcd(precise_time.tm_mday))
    bus.write_byte_data(address, 0x05, dec_to_bcd(precise_time.tm_mon))
    bus.write_byte_data(address, 0x06, dec_to_bcd(precise_time.tm_year % 100))  # Year in two digits
    print("RTC module set successfully to: " + str(precise_time))
    print(time.time())

# Call the function to set the RTC time
set_rtc_time()
