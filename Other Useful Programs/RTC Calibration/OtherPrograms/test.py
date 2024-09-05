import time

from pynput.mouse import Listener


def on_click(x, y, button, pressed):
	if pressed:
		print(f"Mouse clicked at ({x}, {y}) with {button} at {time.time()}")


print("Starting the mouse listener...")

try:
	with Listener(on_click=on_click) as listener:
		print("Listener is now running...")
		listener.join()
except KeyboardInterrupt:
	print("Program interrupted by user.")
except Exception as e:
	print(f"An error occurred: {e}")
finally:
	print("Exiting the program.")
