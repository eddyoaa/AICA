import cv2
import requests
import base64
import numpy as np
import serial
import threading
import keyboard
import time
import os
from datetime import datetime


from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
import threading
from PIL import Image
import numpy as np
import time
from pythonosc import udp_client
# AUDIO CONFIG
# OSC Settings
python_ip = "127.0.0.1"  # IP for Python's OSC server
python_port = 57121      # Port for Python's OSC server
sc_ip = "127.0.0.1"      # SuperCollider IP (localhost)
sc_port = 57120          # SuperCollider OSC port

# Configuration
WEBUI_SERVER_URL = 'http://127.0.0.1:7860'
# COM_PORT = '/dev/tty.usbserial-2110' 
COM_PORT = "/dev/cu.SLAB_USBtoUART"
BAUD_RATE = 9600
WINDOW_NAME = "Display"
DEFAULT_DIMENSION = 512
WINDOW_WIDTH = DEFAULT_DIMENSION * 2
WINDOW_HEIGHT = DEFAULT_DIMENSION
ENTER_TIMEOUT = 1.0  # 1-second timeout for Enter key

# SD Config
# GEN_STEPS = 8
DENOISING_STRENGTH = 0.35  
CFG_SCALE = 8          
SAMPLER = "Euler a"       
MODEL = "SSD-1B"      
PROMPTS = [
    "happy person",
    "happy, smiling person",
    "happy, smiling person",
    "happy, smiling person",
    "person",
    "person",
    "person",
    "unrecognizable, generic person, no facial features",
    "unrecognizable, generic person, no facial features",
    "unrecognizable, generic person, no facial features",
    "unrecognizable, sad, silhouette-like person, no facial features",
    "unrecognizable, sad, silhouette-like person, no facial features",
    "unrecognizable, sad, silhouette-like person, no facial features",
    "faceless, sad sillhouette-like person, no facial features, individuals not distinguishable",
    "faceless, sad sillhouette-like person, no facial features, individuals not distinguishable",
    "faceless, sad sillhouette-like person, no facial features, individuals not distinguishable",
    "faceless, sad sillhouette-like person, no facial features, individuals not distinguishable",
    "circle"
]
PROMPT_ENHANCERS = ", selfie, dystopian, faint, busy background with huge group of people, moody"

LOOP_STEPS = len(PROMPTS)

# ----- General  Setup -----

# Global State Variables
generated_images = []
current_image_index = 0
blend_factor = 0
total_steps = 0
last_enter_time = 0  # Track last 'Enter' press time
CURRENT_WEBCAM_CAPTURE = None  # Store current webcam image

# threading Variables
current_thread : threading.Thread = None
stop_event = threading.Event()

# Webcam Global Constant
WEBCAM = cv2.VideoCapture(0)

# Serial Connection
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    debug_mode = False  # Serial connection exists; debug mode off
    print("Serial connection established")
except serial.SerialException:
    ser = None
    debug_mode = True  # No serial connection; debug mode on
    print("No serial connection available - Debug mode enabled")

# ----- Audio Setup -----

# Function triggered by SuperCollider's OSC message
def evaluate_code(address, *args):
    print(f"Received OSC message: {address} with arguments {args}")
    image_path = 'captured_image.png'
    process_and_send_image(image_path)
    
def process_and_send_image(image_path):
    print("Processing image...")

    # Load and process the image
    image = Image.open(image_path).resize((100, 100))
    image_data = np.array(image) / 255.0  # Normalize to 0-1 range
    
    print(image_data.shape)

    # Separate color channels
    red_channel = image_data[:, :, 0].flatten()
    green_channel = image_data[:, :, 1].flatten()
    blue_channel = image_data[:, :, 2].flatten()

    # Send data via OSC
    main_client.send_message("/image/red", red_channel.tolist())
    main_client.send_message("/image/green", green_channel.tolist())
    main_client.send_message("/image/blue", blue_channel.tolist())

    print("Data sent to SuperCollider!")

# Function to send a stop and reset message
def stop_and_reset():
    # Send a custom OSC message to the SuperCollider listener to stop all processes
    stop_client.send_message("/evaluate/code", None)  # Trigger the block reset in SuperCollider
    print("Sent stop and reset command to SuperCollider.")

# AUDIO SETUP
# Initialize OSC client for sending data to SuperCollider
main_client = SimpleUDPClient(sc_ip, sc_port)
stop_client = SimpleUDPClient(sc_ip, 57123)
# Setup OSC client
rotation_update_client = udp_client.SimpleUDPClient("127.0.0.1", 57122)  # Adjust IP and port as needed
# Set up the dispatcher
dispatcher = Dispatcher()
dispatcher.map("/trigger/evaluate", evaluate_code)  # Map OSC message to function

# Set up the server
server = BlockingOSCUDPServer((python_ip, python_port), dispatcher)

# Create a new thread for the server to run
def server_thread():
    server.serve_forever()

server_thread_instance = threading.Thread(target=server_thread)
server_thread_instance.start()

print(f"Python OSC server running at {python_ip}:{python_port}. Waiting for SuperCollider trigger...")

# ----- Utility Functions -----

def encode_file_to_base64(file_path):
    """Encodes a file to Base64 format."""
    with open(file_path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def decode_base64_to_image(base64_data):
    """Decodes a Base64 string to a NumPy image array."""
    image_bytes = base64.b64decode(base64_data)
    np_array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

def send_image_to_api(image_path, images_placeholder):
    """Sends an image to the API and updates the placeholder with response images."""
    encoded_image = encode_file_to_base64(image_path)
    payload = {
        "prompt": "",
        "init_images": [encoded_image],
        "denoising_strength": 0.2,
        "n_iter": 1,
        "width": DEFAULT_DIMENSION,
        "height": DEFAULT_DIMENSION,
        "batch_size": 1,
        "override_settings": {"sd_model_checkpoint": "v2-1_768-ema-pruned"},
        "script_name": "Loopback",
        "script_args": [10, 0.2, "Linear", "None"]
    }
    response = requests.post(f'{WEBUI_SERVER_URL}/sdapi/v1/img2img', json=payload)
    if response.status_code == 200:
        images = response.json().get("images", [])
        # Pop first image (Loop overview)
        images.pop(0)
        # Decode and add images to placeholder
        images_placeholder.extend(decode_base64_to_image(img) for img in images)
    else:
        print(f"API error: {response.status_code} - {response.text}")

def crop_to_square(image):
    """Crop the center square from an image."""
    h, w = image.shape[:2]
    min_dim = min(h, w)
    x_center, y_center = w // 2, h // 2
    half_dim = min_dim // 2
    return image[y_center - half_dim:y_center + half_dim, x_center - half_dim:x_center + half_dim]


def capture_webcam_image():
    """Capture an image from the webcam."""
    global CURRENT_WEBCAM_CAPTURE

    ret, frame = WEBCAM.read()
    if not ret:
        print("Failed to capture frame.")
        return None, None

    square_image = crop_to_square(frame)
    resized_image = cv2.resize(square_image, (DEFAULT_DIMENSION, DEFAULT_DIMENSION))
    # Save images with a timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    current_image_path = os.path.join("imgs", f"iteration_{timestamp}_{0}.png")
    cv2.imwrite(current_image_path, resized_image)

    # Update the global webcam image
    CURRENT_WEBCAM_CAPTURE = resized_image

    return resized_image, current_image_path


def blend_images(image1, image2, blend_factor):
    """Blend two images."""
    return cv2.addWeighted(image1, 1 - blend_factor, image2, blend_factor, 0)

def main_loop():
    global current_image_index, blend_factor, CURRENT_WEBCAM_CAPTURE

    if not WEBCAM.isOpened():
        print("Could not open webcam.")
        return

    generated_images = []  # List to hold generated and captured images
    current_image_index = 0
    blend_factor = 0
    debug_mode = False
    total_steps = 0
    print("Waiting for serial input...")

    def on_press(key):
        nonlocal debug_mode, total_steps
        try:
            if hasattr(key, 'char') and key.char == 'D':  # Großes D bedeutet Shift+d
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            
            if debug_mode:
                if key == keyboard.Key.enter:
                    # Simulate button press
                    print(f"Enter pressed: Captured image")
                    captured_image, image_path = capture_webcam_image(cap, DEFAULT_DIMENSION)
                    if captured_image is not None:
                        generated_images.clear()
                        generated_images.append(captured_image)
                        threading.Thread(target=send_image_to_api, args=(image_path, generated_images)).start()
                        current_image_index = 0
                
                elif key == keyboard.Key.left:
                    total_steps = max(0, total_steps - 1)
                    print(f"Steps: {total_steps}")
                
                elif key == keyboard.Key.right:
                    total_steps = min(39, total_steps + 1)  # Max value would be 39 (2 images * 20 steps - 1)
                    print(f"Steps: {total_steps}")

        except AttributeError:
            pass

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while True:
        # Handle inputs
        if debug_mode:
            handle_keyboard_input()
        elif ser and ser.in_waiting > 0:
            if not handle_serial_input():
                break

        # Capture webcam frame
        ret, webcam_frame = WEBCAM.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        current_webcam_frame = crop_to_square(webcam_frame)

        # Create split screen with blended image, if available
        if len(generated_images) > 1:
            # image blending
            blend_factor = (total_steps % 20) / 19.0
            current_image_index = min(total_steps // 20, len(generated_images) - 2)
            blended_image = blend_images(
                generated_images[current_image_index],
                generated_images[current_image_index + 1],
                blend_factor
            )
            # audio blending
            if rotation_update_client:
                rotation_update_client.send_message("/knob", [current_image_index, blend_factor])
        elif CURRENT_WEBCAM_CAPTURE is not None:
            blended_image = CURRENT_WEBCAM_CAPTURE
        else:
            blended_image = current_webcam_frame

        split_screen = create_split_screen(current_webcam_frame, blended_image)
        cv2.imshow(WINDOW_NAME, split_screen)

        # Check for window close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            stop_event.set()
            break

    if ser:
        ser.close()
    WEBCAM.release()
    # AUDIO shutdown
    stop_and_reset()
    # Shutdown server gracefully after the task is done
    server.shutdown()
    server_thread_instance.join()
    print("Server has been stopped.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()