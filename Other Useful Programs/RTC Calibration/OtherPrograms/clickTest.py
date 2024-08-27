from pynput.mouse import Listener
import logging
import datetime
import time

logging.basicConfig(filename="mouse_log.txt", level=logging.INFO, format='%(asctime)s: %(message)s')

def on_click(x, y, button, pressed):
	if pressed:
		logging.info(f"Mouse clicked at ({x}, {y}) with {button} at {time.time()}")


with Listener(on_click=on_click) as listener:
	listener.join()
