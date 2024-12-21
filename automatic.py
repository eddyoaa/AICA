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