import cv2
import requests
import base64
import numpy as np
import serial
import threading
import keyboard
import time

# Configuration
WEBUI_SERVER_URL = 'http://127.0.0.1:7860'
COM_PORT = '/dev/tty.usbserial-2110'
BAUD_RATE = 9600
WINDOW_NAME = "Display"
DEFAULT_DIMENSION = 512
WINDOW_WIDTH = DEFAULT_DIMENSION * 2
WINDOW_HEIGHT = DEFAULT_DIMENSION
ENTER_TIMEOUT = 1.0  # 1-second timeout for Enter key

# SD Config
GEN_STEPS = 5
DENOISING_STRENGTH = 0.3  # Balanced preservation and transformation
CFG_SCALE = 8.5           # Lower for faster and more organic results
SAMPLER = "Euler a"       # Fast and reliable sampler
MODEL = "SSD-1B"      # Use a lightweight model for speed
PROMPTS = [
    "happy, smiling person",
    "happy, smiling person",
    "happy, smiling person in isolation",
    "person in huge group of people",
    "anonymous, silhouette-like person, no facial features, in huge group of people",
    "anonymous, silhouette-like person, no facial features, in huge group of people",
    "anonymous, silhouette-like person, no facial features, in huge group of people",
    "anonymous, sad, silhouette-like person, no facial features, in huge group of people",
    "anonymous, sad, silhouette-like person, no facial features, in huge group of people",
    "anonymous, sad, silhouette-like person, no facial features, in huge group of people",
    "no facial features, huge group of people, individuals not distinguishable",
    "no facial features, huge group of people, individuals not distinguishable",
    "no facial features, huge group of people, individuals not distinguishable",
    "no facial features, huge group of people, individuals not distinguishable",
    "no facial features, huge group of people, individuals not distinguishable"
]
PROMPT_ENHANCERS = ", dramatic shadows, busy background, foggy, moody, realistic"

LOOP_STEPS = len(PROMPTS)


# Global State Variables
generated_images = []
current_image_index = 0
blend_factor = 0
total_steps = 0
last_enter_time = 0  # Track last 'Enter' press time
CURRENT_WEBCAM_CAPTURE = None  # Store current webcam image

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


def iterative_image_generation(image_path, generated_imgs : list):
    """
    Performs iterative image generation by sending successive API requests.
    Each iteration uses the output of the previous one as the new input.
    """
    current_image_path = image_path
    for i in range(LOOP_STEPS):
        encoded_image = encode_file_to_base64(current_image_path)
        # API Payload
        payload = {
            "prompt": PROMPTS[i] + PROMPT_ENHANCERS,
            "negative_prompt": "flat, ugly, boring, low quality",
            "init_images": [encoded_image],  # Base image for img2img
            "denoising_strength": DENOISING_STRENGTH,
            "cfg_scale": CFG_SCALE,
            "sampler_index": SAMPLER,
            "steps": GEN_STEPS,
            "width": DEFAULT_DIMENSION,
            "height": DEFAULT_DIMENSION,
            "n_iter": 1,  # Number of times to run
            "batch_size": 1,  # Process one image at a time for speed
            "override_settings": {
                "sd_model_checkpoint": MODEL,
                "CLIP_stop_at_last_layers": 2  # Optimize text-to-image processing
            },
        }
        response = requests.post(f'{WEBUI_SERVER_URL}/sdapi/v1/img2img', json=payload)
        if response.status_code == 200:
            images = response.json().get("images", [])
            if not images:
                print("No images received from API.")
                break
            
            new_image = decode_base64_to_image(images[0])
            generated_imgs.append(new_image)

            current_image_path = "current_iteration.png"
            cv2.imwrite(current_image_path, new_image)
        else:
            print(f"API error during iteration: {response.status_code} - {response.text}")
            break


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
    image_path = 'captured_image.png'
    cv2.imwrite(image_path, resized_image)

    # Update the global webcam image
    CURRENT_WEBCAM_CAPTURE = resized_image

    return resized_image, image_path


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
                generated_images.clear()
                generated_images.append(captured_image)
                threading.Thread(target=iterative_image_generation, args=(image_path, generated_images)).start()
                current_image_index = 0
        else:
            print("Enter pressed too soon, ignoring duplicate.")

    elif command == "left":
        total_steps = max(0, total_steps - 1)
        print(f"Steps: {total_steps}")

    elif command == "right":
        if len(generated_images) > 0:
            n_images = len(generated_images) -1
            total_steps = min(n_images * 20 - 1, total_steps + 1)
        else:
            total_steps = 0
        print(f"Steps: {total_steps}")

    elif command.isdigit():
        total_steps = int(command)
        current_image_index = total_steps // 20
        print(f"Set steps to: {total_steps}")


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
        if value:
            handle_command(value)
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
            blend_factor = (total_steps % 20) / 19.0
            current_image_index = min(total_steps // 20, len(generated_images) - 2)
            blended_image = blend_images(
                generated_images[current_image_index],
                generated_images[current_image_index + 1],
                blend_factor
            )
        elif CURRENT_WEBCAM_CAPTURE is not None:
            blended_image = CURRENT_WEBCAM_CAPTURE
        else:
            blended_image = current_webcam_frame

        split_screen = create_split_screen(current_webcam_frame, blended_image)
        cv2.imshow(WINDOW_NAME, split_screen)

        # Check for window close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    if ser:
        ser.close()
    WEBCAM.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()