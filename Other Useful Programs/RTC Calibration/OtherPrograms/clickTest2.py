from pynput.mouse import Listener
import datetime
import time

def on_click(x, y, button, pressed):
    if pressed:
        print(f"Mouse clicked at ({x}, {y}) with {button} at {time.time()}")

with Listener(on_click=on_click) as listener:
    listener.join()

