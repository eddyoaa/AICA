import cv2
import keyboard
import requests
import base64
import json
import numpy as np
import serial
import time

# Define the URL for the img2img API
webui_server_url = 'http://127.0.0.1:7860'

# Konfigurieren Sie den COM-Port und die Baudrate
COM_PORT = 'COM15'  # Ändern Sie dies auf den richtigen COM-Port
BAUD_RATE = 9600

# Serielle Verbindung einrichten
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

def encode_file_to_base64(path):
    print(f"Kodieren der Datei {path} in Base64...")
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def display_image(image_data):
    # Dekodieren des Base64-Bildes
    try:
        image_bytes = base64.b64decode(image_data)
        # Konvertieren der Bytes in ein NumPy-Array
        np_array = np.frombuffer(image_bytes, np.uint8)
        # Konvertieren des NumPy-Arrays in ein Bild
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            print("Fehler beim Dekodieren des Bildes. Das Bild könnte leer oder ungültig sein.")
            return

        # Bild anzeigen
        cv2.imshow('Generated Image', image)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Wenn 'q' gedrückt wird, schließen
                break
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Fehler beim Anzeigen des Bildes: {e}")

def send_image(image_path):
    print(f"Bild wird gesendet: {image_path}")
    encoded_image = encode_file_to_base64(image_path)

    payload = {
        "prompt": "1girl, blue hair",  # Beispiel-Prompt
        "init_images": [encoded_image],  # Das kodierte Bild zur Payload hinzufügen
        "denoising_strength": 0.5,
        "n_iter": 1,
        "width": 256,
        "height": 256,
        "batch_size": 1,
    }

    print("Sende Payload an die API...")
    response = requests.post(f'{webui_server_url}/sdapi/v1/img2img', json=payload)
    
    if response.status_code == 200:
        print("Bild erfolgreich gesendet!")
        r = response.json()
        print("Antwort von der API erhalten:")
        print(json.dumps(r, indent=4))  # Schöner formatierte Ausgabe der Antwort
        
        # Bild anzeigen
        if "images" in r and len(r["images"]) > 0:
            print("Bilddaten:", r["images"][0])  # Debug-Ausgabe der Bilddaten
            display_image(r["images"][0])  # Zeige das erste Bild an
    else:
        print(f"Fehler beim Senden des Bildes: {response.status_code} - {response.text}")

def capture_image():
    print("Öffne Webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Konnte die Webcam nicht öffnen.")
        return

    # Frame von der Webcam lesen
    ret, frame = cap.read()
    if not ret:
        print("Konnte das Frame nicht lesen.")
        return

    # Frame anzeigen
    cv2.imshow('Webcam', frame)

    # Überprüfen, ob die Eingabetaste gedrückt wurde
    # Bild speichern
    image_path = 'captured_image.png'
    cv2.imwrite(image_path, frame)
    print("Bild gespeichert als 'captured_image.png'")
    cap.release()
    cv2.destroyAllWindows()
    send_image(image_path)  # Bild an die API senden

    # Ressourcen freigeben
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam geschlossen und Ressourcen freigegeben.")

def main():
    print("Warten auf serielle Eingabe...")
    while True:
        if ser.in_waiting > 0:  # Überprüfen, ob Daten verfügbar sind
            value = str(ser.readline().rstrip()).replace("'","").replace("b","",1) # Wert lesen
            print(value)
            if(value=="btn"):
                print(f"Wert empfangen: {value}")
                capture_image()  # Bild aufnehmen, wenn ein Wert empfangen wird
                time.sleep(1)  # Kurze Pause, um Mehrfachaufnahmen zu vermeiden

if __name__ == "__main__":
    main()