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

def iterative_image_generation(image_path, generated_imgs: list):
    """
    Stops the existing thread (if running) and starts a new instance of the iterative image generation function.
    """
    global stop_event, current_thread

    # Stop the current thread if it's running
    if current_thread and current_thread.is_alive():
        print("Stopping the currently running thread.")
        stop_event.set()  # Signal the thread to stop
        current_thread.join()  # Wait for it to finish
        print("Thread stopped.")

    # Reset Stop event
    stop_event.clear()

    # Create and start a new thread
    new_thread = threading.Thread(
        target=_iterative_image_generation,
        args=(image_path, generated_imgs)
    )
    new_thread.start()

    # Update thread manager with new thread and stop event
    current_thread = new_thread
    stop_event = stop_event

    print("New thread started.")



def _iterative_image_generation(image_path, generated_imgs : list):
    """
    Performs iterative image generation by sending successive API requests.
    Each iteration uses the output of the previous one as the new input.
    """
    current_image_path = image_path
    for i in range(LOOP_STEPS):
        if stop_event.is_set():
            return
        encoded_image = encode_file_to_base64(current_image_path)
        # API Payload
        payload = {
            "prompt": PROMPTS[i] + PROMPT_ENHANCERS,
            "negative_prompt": "flat, ugly, boring, low quality, blurry",
            "init_images": [encoded_image],  # Base image for img2img
            "denoising_strength": DENOISING_STRENGTH,
            "cfg_scale": CFG_SCALE,
            "sampler_index": SAMPLER,
            "steps": max(2, min(8, LOOP_STEPS - i)), # Reduce step size in every loop, ranging from 2 to 8
            "width": DEFAULT_DIMENSION,
            "height": DEFAULT_DIMENSION,
            "n_iter": 1,  # Number of times to run
            "batch_size": 1,  # Process one image at a time for speed
            "override_settings": {
                "sd_model_checkpoint": MODEL,
                "CLIP_stop_at_last_layers": 2
            },
        }
        try:
            response = requests.post(f'{WEBUI_SERVER_URL}/sdapi/v1/img2img', json=payload, timeout=10)  # Set timeout to 10 seconds
            if response.status_code == 200:
                images = response.json().get("images", [])
                if not images:
                    print("No images received from API.")
                    break
                if stop_event.is_set():
                    return

                new_image = decode_base64_to_image(images[0])
                generated_imgs.append(new_image)

                # Save images with a timestamp to avoid overwriting
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                current_image_path = os.path.join("imgs", f"iteration_{timestamp}_{i+1}.png")
                cv2.imwrite(current_image_path, new_image)
            else:
                print(f"API error during iteration {i}: {response.status_code} - {response.text}")
                break
        except requests.exceptions.Timeout:
            print(f"Timeout occurred during iteration {i}. Debug: The request took longer than 10 seconds.")

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


def create_split_screen(webcam_frame, blended_image):
    """Create a side-by-side split screen view."""
    side_width = WINDOW_WIDTH // 2
    webcam_resized = cv2.resize(webcam_frame, (side_width, WINDOW_HEIGHT))
    blended_resized = cv2.resize(blended_image, (side_width, WINDOW_HEIGHT))
    return cv2.hconcat([webcam_resized, blended_resized])


# ----- Command Handler -----

def handle_command(command):
    """Process commands from keyboard or serial input."""
    global total_steps, current_image_index, last_enter_time

    current_time = time.time()
    if command == "btn" or command == "enter":
        if current_time - last_enter_time > ENTER_TIMEOUT:
            last_enter_time = current_time
            print("Command: Capture Image")
            captured_image, image_path = capture_webcam_image()
            if captured_image is not None:
                main_client.send_message("/evaluate/code", [])
                generated_images.clear()
                generated_images.append(captured_image)
                current_image_index = 0
                total_steps = 0
                iterative_image_generation(image_path, generated_images)
        else:
            print("Enter pressed too soon, ignoring duplicate.")

    elif command == "left":
        total_steps = max(0, total_steps - 1)
        print(f"Steps: {total_steps}")

    elif command == "right":
        if len(generated_images) > 1:
            n_images = len(generated_images) -1
            total_steps = min(n_images * 20 - 1, total_steps + 1)
        else:
            total_steps = 0
        print(f"Steps: {total_steps}")

    # elif command.isdigit():
    #     total_steps = int(command)
    #     current_image_index = total_steps // 20
    #     print(f"Set steps to: {total_steps}")
    else:
        print(f"unable to decode command {command}")

# ----- Input Handlers -----

def handle_keyboard_input():
    """Check for keyboard inputs and delegate commands."""
    if keyboard.is_pressed('enter'):
        handle_command("enter")
    if keyboard.is_pressed('left'):
        handle_command("left")
    if keyboard.is_pressed('right'):
        handle_command("right")


def handle_serial_input():
    """Handle serial input and delegate commands."""
    try:
        value = ser.readline().decode('utf-8').strip()
        if value == "btn":
            handle_command("enter")
        elif value == "+":
            handle_command("left")
        elif value == "-":
            handle_command("right")
    except serial.SerialException:
        print("Serial connection lost")
        return False
    return True


# ----- Main Loop -----

def main_loop():
    global current_image_index, blend_factor, CURRENT_WEBCAM_CAPTURE

    if not WEBCAM.isOpened():
        print("Could not open webcam.")
        return

    print("Waiting for input...")

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
import cv2
import requests
import base64
import numpy as np
import serial
import threading
from pynput import keyboard
from PIL import Image
import win32print
import win32ui
from io import BytesIO

# Configuration
WEBUI_SERVER_URL = 'http://127.0.0.1:7860'
COM_PORT = '/dev/tty.usbserial-2110'
BAUD_RATE = 9600
WINDOW_NAME = "Display"
DEFAULT_DIMENSION = 512  # Default square dimension for the webcam and images
WINDOW_WIDTH = DEFAULT_DIMENSION * 2
WINDOW_HEIGHT = DEFAULT_DIMENSION

# Try to set up serial connection
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    serial_available = True
    print("Serial connection established")
except serial.SerialException:
    ser = None
    serial_available = False
    print("No serial connection available - continuing in keyboard-only mode")

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
        "script_args": [1, 0.2, "Linear", "None"]
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
    """Crops an image to a square from the center."""
    h, w = image.shape[:2]
    min_dim = min(h, w)
    x_center = w // 2
    y_center = h // 2
    half_dim = min_dim // 2
    return image[y_center - half_dim:y_center + half_dim, x_center - half_dim:x_center + half_dim]

def capture_webcam_image(cap, dimension):
    """Captures and processes an image from the webcam."""
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        return None, None
    cropped_square = crop_to_square(frame)
    resized_image = cv2.resize(cropped_square, (dimension, dimension))
    image_path = 'captured_image.png'
    cv2.imwrite(image_path, resized_image)
    return resized_image, image_path

def create_split_screen(webcam_frame, blended_image, width, height):
    """Creates a split-screen view of the webcam and blended image."""
    side_width = width // 2
    webcam_resized = cv2.resize(webcam_frame, (side_width, height))
    blended_resized = cv2.resize(blended_image, (side_width, height))
    return cv2.hconcat([webcam_resized, blended_resized])

def blend_images(image1, image2, blend_factor):
    """Blends two images using the given blend factor."""
    return cv2.addWeighted(image1, 1 - blend_factor, image2, blend_factor, 0)

def print_image(image):
    """Druckt ein OpenCV-Bild auf dem Standarddrucker."""
    try:
        # OpenCV Bild zu PIL Image konvertieren
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Standarddrucker ermitteln
        printer_name = win32print.GetDefaultPrinter()
        
        # Druckauftrag starten
        hprinter = win32print.OpenPrinter(printer_name)
        print(f"Drucke auf: {printer_name}")
        
        # Hier können Sie die Druckeinstellungen anpassen
        pil_image.save("temp_print.png")
        win32print.StartDocPrinter(hprinter, 1, ("temp_print.png", None, "RAW"))
        win32print.EndDocPrinter(hprinter)
        win32print.ClosePrinter(hprinter)
        
    except Exception as e:
        print(f"Druckfehler: {str(e)}")

def main_loop():
    """Main loop to handle serial/keyboard input and process images."""
    global serial_available, ser
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
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
                
                elif hasattr(key, 'char') and key.char == 'p':
                    if len(generated_images) > 0:
                        current_image = blend_images(
                            generated_images[current_image_index],
                            generated_images[current_image_index + 1],
                            blend_factor
                        ) if len(generated_images) > 1 else generated_images[0]
                        print_image(current_image)
                        print("Druckauftrag gesendet")
                
        except AttributeError:
            pass

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while True:
        # Read the webcam frame
        ret, webcam_frame = cap.read()
        webcam_frame = crop_to_square(webcam_frame)
        if not ret:
            print("Failed to read frame from webcam.")
            break
        
        if len(generated_images) > 1:
            # Calculate blend factor
            if debug_mode:
                current_image_index = total_steps // 20
                blend_factor = (total_steps % 20) / 20
                current_image_index %= len(generated_images) - 1
            blended_image = blend_images(generated_images[current_image_index], generated_images[current_image_index + 1], blend_factor)
        elif len(generated_images) == 1:
            blended_image = generated_images[0]
        else:
            blended_image = webcam_frame  # Display the webcam frame if no generated images

        # Display the split screen with webcam and blended image
        split_screen = create_split_screen(webcam_frame, blended_image, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.imshow(WINDOW_NAME, split_screen)

        # Check for input from ESP32 if serial is available
        if serial_available and ser.in_waiting > 0:
            try:
                value = ser.readline().decode('utf-8').strip()
                print(f"Received value: {value}")

                if value == "btn":
                    captured_image, image_path = capture_webcam_image(cap, DEFAULT_DIMENSION)
                    if captured_image is not None:
                        generated_images.clear()
                        generated_images.append(captured_image)
                        threading.Thread(target=send_image_to_api, args=(image_path, generated_images)).start()
                        current_image_index = 0

                if value.isdigit() and len(generated_images) > 1:
                    total_steps = int(value)
                    current_image_index = total_steps // 20
                    blend_factor = (total_steps % 20) / 20
                    current_image_index %= len(generated_images) -1
            except serial.SerialException:
                print("Serial connection lost")
                serial_available = False
                ser = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Cleanup
    listener.stop()
    if serial_available:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()