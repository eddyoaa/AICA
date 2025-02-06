from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
import threading
from PIL import Image
import numpy as np
import time
import tkinter as tk
import math
from pythonosc import udp_client

# OSC Settings
python_ip = "127.0.0.1"  # IP for Python's OSC server
python_port = 57121      # Port for Python's OSC server
sc_ip = "127.0.0.1"      # SuperCollider IP (localhost)
sc_port = 57120          # SuperCollider OSC port

# Initialize OSC client for sending data to SuperCollider
client = SimpleUDPClient(sc_ip, sc_port)

client.send_message("/evaluate/code", [])
print("Evaluation signal sent!")


# Function to process the image and send data to SuperCollider
def process_and_send_image(image_path):
    print("Processing image...")

    # Load and process the image
    image = Image.open(image_path).resize((100, 100))
    image_data = np.array(image) / 255.0  # Normalize to 0-1 range

    # Separate color channels
    red_channel = image_data[:, :, 0].flatten()
    green_channel = image_data[:, :, 1].flatten()
    blue_channel = image_data[:, :, 2].flatten()

    # Send data via OSC
    client.send_message("/image/red", red_channel.tolist())
    client.send_message("/image/green", green_channel.tolist())
    client.send_message("/image/blue", blue_channel.tolist())

    print("Data sent to SuperCollider!")


# Function triggered by SuperCollider's OSC message
def evaluate_code(address, *args):
    print(f"Received OSC message: {address} with arguments {args}")
    image_path = "C:/Users/Tukora Péter/Pictures/kunsthallekarlsruhe-willem-van-aelst-stillleben-mit-jagdgeraeten-und-totem-re-350.small_.jpg"
    process_and_send_image(image_path)

    # After processing and sending data, stop the server
    stop_event.set()  # Set the event to signal the server to stop


# Set up the dispatcher
dispatcher = Dispatcher()
dispatcher.map("/trigger/evaluate", evaluate_code)  # Map OSC message to function

# Set up the server
server = BlockingOSCUDPServer((python_ip, python_port), dispatcher)

# Event to signal server to stop
stop_event = threading.Event()

# Create a new thread for the server to run
def server_thread():
    server.serve_forever()

server_thread_instance = threading.Thread(target=server_thread)
server_thread_instance.start()

print(f"Python OSC server running at {python_ip}:{python_port}. Waiting for SuperCollider trigger...")

# Wait for the stop event to be set (i.e., stop the server after processing)
stop_event.wait()

# Shutdown server gracefully after the task is done
server.shutdown()
server_thread_instance.join()

print("Server has been stopped.")


class Knob(tk.Canvas):
    def __init__(self, parent, size=100, osc_client=None, **kwargs):
        super().__init__(parent, width=size, height=size, **kwargs)
        self.size = size
        self.angle = 0
        self.total_angle = 0  # Track total rotation
        self.center = size // 2
        self.knob_radius = size // 2 - 10
        self.max_turns = 10  # Maximum number of full turns allowed
        self.max_total_angle = 360 * self.max_turns  # 360 degrees per turn
        self.osc_client = osc_client  # OSC client

        # Draw the knob
        self.knob = self.create_oval(10, 10, size - 10, size - 10, fill="gray")
        self.indicator = self.create_line(self.center, self.center,
                                          self.center, 10, width=3, fill="red")

        # Mouse event bindings
        self.bind("<Button-1>", self.start_rotate)
        self.bind("<B1-Motion>", self.rotate)

    def start_rotate(self, event):
        self.last_angle = self._calculate_angle(event.x, event.y)

    def rotate(self, event):
        new_angle = self._calculate_angle(event.x, event.y)
        delta_angle = new_angle - self.last_angle

        # Handle the crossing of 0-degree boundary
        if delta_angle < -180:
            delta_angle += 360
        elif delta_angle > 180:
            delta_angle -= 360

        # Update total angle within bounds of 0 to max_total_angle
        new_total_angle = self.total_angle + delta_angle
        if 0 <= new_total_angle <= self.max_total_angle:
            self.total_angle = new_total_angle
            self.angle = self.total_angle % 360
            self.last_angle = new_angle

            # Calculate the current turn count and fractional position
            current_turn = int(self.total_angle // 360)
            fractional_turn = (self.total_angle % 360) / 360.0

            # Send the current turn count and fractional position via OSC
            if self.osc_client:
                self.osc_client.send_message("/knob", [current_turn, fractional_turn])

            # Rotate the indicator
            self._update_indicator()

    def _calculate_angle(self, x, y):
        """Calculate the angle between the center and the point (x, y)."""
        dx = x - self.center
        dy = y - self.center
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def _update_indicator(self):
        """Update the position of the indicator based on the current angle."""
        radians = math.radians(self.angle)
        x_end = self.center + self.knob_radius * math.cos(radians)
        y_end = self.center + self.knob_radius * math.sin(radians)
        self.coords(self.indicator, self.center, self.center, x_end, y_end)




# Create the main window
root = tk.Tk()
root.title("Knob Control")

# Setup OSC client
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 57122)  # Adjust IP and port as needed

# Create and place the knob
knob = Knob(root, size=150, osc_client=osc_client)
knob.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()