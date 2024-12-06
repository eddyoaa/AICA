import tkinter as tk
from tkinter import messagebox
import base64
import io
import cv2
import numpy as np

def show_image():
    base64_string = entry.get("1.0", tk.END).strip()  # Text aus dem Textfeld abrufen
    try:
        # Dekodieren des Base64-Strings
        image_data = base64.b64decode(base64_string)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)  # Bild aus Bytes erstellen

        # Bild anzeigen
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler beim Anzeigen des Bildes: {e}")

# Hauptfenster erstellen
root = tk.Tk()
root.title("Base64 Bildanzeige")

# Textfeld für Base64-String
entry = tk.Text(root, height=10, width=50)
entry.pack(pady=10)

# Button zum Anzeigen des Bildes
show_button = tk.Button(root, text="Bild anzeigen", command=show_image)
show_button.pack(pady=5)

# Hauptschleife starten
root.mainloop()
