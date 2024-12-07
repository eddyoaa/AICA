import cv2
import requests
import base64
import numpy as np
import serial
import time
import threading

# Configuration
WEBUI_SERVER_URL = 'http://127.0.0.1:7860'
COM_PORT = '/dev/tty.usbserial-2110'
BAUD_RATE = 9600
WINDOW_NAME = "Display"
DEFAULT_DIMENSION = 512  # Default square dimension for the webcam and images
WINDOW_WIDTH = DEFAULT_DIMENSION * 2
WINDOW_HEIGHT = DEFAULT_DIMENSION

# Set up serial connection
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

def encode_file_to_base64(file_path):
    """Encodes a file to Base64 format."""
    with open(file_path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def decode_base64_to_image(base64_data):
    """Decodes a Base64 string to a NumPy image array."""
    try:
        image_bytes = base64.b64decode(base64_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

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
        # pop first image (Loop overview)
        images.pop(0)
        images_placeholder[:] = [decode_base64_to_image(img) for img in images if decode_base64_to_image(img) is not None]
    else:
        print(f"Error from API: {response.status_code} - {response.text}")

def crop_to_square(image):
    """Crops an image to a square from the center."""
    h, w = image.shape[:2]
    min_dim = min(h, w)
    x_center = w // 2
    y_center = h // 2
    half_dim = min_dim // 2
    return image[y_center - half_dim:y_center + half_dim, x_center - half_dim:x_center + half_dim]

def capture_webcam_image(cap, dimension):
    """Captures a square-cropped image from the webcam and saves it to a file."""
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        return None

    # Crop the frame to a square
    cropped_frame = crop_to_square(frame)

    # Resize to the required dimension if needed
    cropped_square = cv2.resize(cropped_frame, (dimension, dimension))

    image_path = 'captured_image.png'
    cv2.imwrite(image_path, cropped_square)
    return image_path

def create_split_screen(webcam_frame, generated_image, width, height):
    """Creates a split-screen view with webcam on the left and generated image on the right."""
    side_width = width // 2

    # Resize webcam image to fit in the left pane
    webcam_resized = cv2.resize(webcam_frame, (side_width, height))

    # Directly resize generated image for the right pane
    generated_resized = cv2.resize(generated_image, (side_width, height))

    # Concatenate the images
    split_screen = cv2.hconcat([webcam_resized, generated_resized])
    return split_screen

def main_loop():
    """Main loop to handle serial input, display webcam feed, and process images."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    generated_images = [np.zeros((DEFAULT_DIMENSION, DEFAULT_DIMENSION, 3), dtype=np.uint8)]  # Placeholder for generated images
    current_image_index = 0
    print("Waiting for serial input...")

    while True:
        # try to read webcam frame
        ret, webcam_frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break
        # crop webcam frame to square
        square_crop = crop_to_square(webcam_frame)
        square_crop_resized = cv2.resize(square_crop, (DEFAULT_DIMENSION, DEFAULT_DIMENSION))
        
        # place webcam frame and generated_image into splitscreen and display
        split_screen = create_split_screen(square_crop_resized, generated_images[current_image_index], WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.imshow(WINDOW_NAME, split_screen)

        # Check for Input from ESP32
        if ser.in_waiting > 0:
            value = ser.readline().decode('utf-8').strip()
            print(f"Received value: {value}")
            # Button Press -> trigger image capture
            if value == "btn":
                # Capture and process square-cropped image
                image_path = capture_webcam_image(cap, DEFAULT_DIMENSION)
                if image_path:
                    # Run API call in a separate thread
                    threading.Thread(target=send_image_to_api, args=(image_path, generated_images)).start()
                    current_image_index = 0  # Reset to the first image

                time.sleep(1)  # Avoid multiple triggers in a short time
            # rotary input -> iterate through generated images
            if value.isdigit():
                # Parse the input number and use it as an index
                index = int(value)
                if generated_images:
                    current_image_index = index % len(generated_images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()