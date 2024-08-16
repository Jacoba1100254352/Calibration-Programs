import time

from pynput.mouse import Listener


# Uncomment the following line if you want to log the events to a file.
# logging.basicConfig(filename="mouse_log.txt", level=logging.INFO, format='%(asctime)s: %(message)s')

def on_click(x, y, button, pressed):
	if pressed:
		print(f"Mouse clicked at ({x}, {y}) with {button} at {time.time()}")
        # Uncomment the following line if you want to log the events to a file.
        # logging.info(f"Mouse clicked at ({x}, {y}) with {button} at {time.time()}")


try:
	with Listener(on_click=on_click) as listener:
		listener.join()
except KeyboardInterrupt:
	print()
