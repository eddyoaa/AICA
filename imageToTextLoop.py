import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import requests
import io
from openai import OpenAI
from pathlib import Path
import numpy as np
import pygame
import pygame.sndarray
import colorsys

class ImageTextLoop:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.current_description = "Ein friedlicher Sonnenuntergang über einem ruhigen See"
        self.current_image_url = None
        self.last_image_url = None
        
        # GUI Setup
        self.root = tk.Tk()
        self.root.title("Bild-Text-Loop")
        
        # Button erstellen
        self.generate_button = tk.Button(self.root, text="Neues Bild generieren", command=self.generate_image)
        self.generate_button.pack(pady=10)
        
        # Label für das Bild
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        self.image_label.bind('<Button-1>', self.analyze_image)
        
        # Label für die Beschreibung
        self.description_label = tk.Label(self.root, wraplength=400, text=self.current_description)
        self.description_label.pack(pady=10)
        
        # Loading-Status Labels hinzufügen
        self.loading_label = tk.Label(self.root, text="", fg="blue")
        self.loading_label.pack(pady=5)
        
        # Button-Text speichern
        self.default_button_text = "Neues Bild generieren"
        
        # Pygame Audio initialisieren
        pygame.mixer.init(44100, -16, 1, 2048)
        self.sample_rate = 44100
        self.duration = 2.0  # Sekunden
        
        # Sound Button hinzufügen
        self.sound_button = tk.Button(self.root, text="🔊 Sound abspielen", command=self.play_current_sound)
        self.sound_button.pack(pady=5)
        self.sound_button.config(state=tk.DISABLED)  # Initial deaktiviert
        
        # Sound-Status speichern
        self.current_sound = None

    def generate_image(self):
        try:
            # Loading-Status anzeigen
            self.loading_label.config(text="Generiere neues Bild...")
            self.generate_button.config(state=tk.DISABLED, text="Wird generiert...")
            self.root.update()
            
            # Prompt mit Referenzbild erstellen, falls vorhanden
            if self.last_image_url:
                response = self.client.images.create_variation(
                    image=requests.get(self.last_image_url).content,
                    n=1,
                    size="256x256"
                )
            else:
                response = self.client.images.generate(
                    model="dall-e-2",
                    prompt=self.current_description,
                    size="256x256",
                    n=1,
                )
            
            # Aktuelles Bild als letztes Bild speichern
            self.last_image_url = self.current_image_url
            # Neue Bild-URL speichern
            self.current_image_url = response.data[0].url
            image_data = requests.get(self.current_image_url).content
            image = Image.open(io.BytesIO(image_data))
            
            # Bild für GUI vorbereiten
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            # Loading-Status zurücksetzen
            self.loading_label.config(text="")
            self.generate_button.config(state=tk.NORMAL, text=self.default_button_text)
            
        except Exception as e:
            print(f"Fehler bei der Bildgenerierung: {e}")
            self.loading_label.config(text="Fehler bei der Bildgenerierung!")
            self.generate_button.config(state=tk.NORMAL, text=self.default_button_text)

    def image_to_sound(self, image):
        """Konvertiert ein Bild in Sound basierend auf Farben und Helligkeit"""
        try:
            # Bild in numpy Array konvertieren
            img_array = np.array(image)
            
            # Durchschnittliche Farben und Helligkeit berechnen
            avg_color = np.mean(img_array, axis=(0,1))
            brightness = np.mean(avg_color)
            
            # RGB zu HSV konvertieren für bessere Tonzuordnung
            hsv = colorsys.rgb_to_hsv(avg_color[0]/255, avg_color[1]/255, avg_color[2]/255)
            
            # Frequenz basierend auf Farbton (100-1000 Hz)
            frequency = 100 + hsv[0] * 900
            
            # Amplitude basierend auf Helligkeit
            amplitude = brightness / 255.0
            
            # Zeitarray erstellen
            t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
            
            # Sinuswelle generieren
            wave = np.sin(2 * np.pi * frequency * t) * amplitude
            
            # In 16-bit Integer konvertieren
            sound_array = (wave * 32767).astype(np.int16)
            
            # Stereo-Array erstellen (2 Kanäle) und C-contiguous machen
            stereo_array = np.column_stack((sound_array, sound_array)).copy()
            
            # Sound erstellen und speichern
            sound = pygame.sndarray.make_sound(stereo_array)
            self.current_sound = sound
            self.sound_button.config(state=tk.NORMAL)  # Button aktivieren
            
        except Exception as e:
            print(f"Fehler bei der Soundgenerierung: {e}")
            self.sound_button.config(state=tk.DISABLED)

    def play_current_sound(self):
        """Spielt den aktuell gespeicherten Sound ab"""
        if self.current_sound:
            self.current_sound.play()

    def analyze_image(self, event):
        if not self.current_image_url:
            print("Kein Bild verfügbar für die Analyse")
            return
            
        try:
            # Loading-Status anzeigen
            self.loading_label.config(text="Analysiere Bild...")
            self.generate_button.config(state=tk.DISABLED)
            self.sound_button.config(state=tk.DISABLED)  # Button während Analyse deaktivieren
            self.root.update()
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Beschreibe dieses Bild detailliert und kreativ für die Generierung eines neuen Bildes."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self.current_image_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            
            self.current_description = response.choices[0].message.content
            self.description_label.config(text=self.current_description)
            
            # Nach der Analyse Sound generieren
            image = Image.open(io.BytesIO(requests.get(self.current_image_url).content))
            self.image_to_sound(image)
            
            # Loading-Status zurücksetzen
            self.loading_label.config(text="")
            self.generate_button.config(state=tk.NORMAL)
            
        except Exception as e:
            print(f"Fehler bei der Bildanalyse: {e}")
            self.loading_label.config(text="Fehler bei der Bildanalyse!")
            self.generate_button.config(state=tk.NORMAL)
            self.sound_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    API_KEY = ""
    app = ImageTextLoop(API_KEY)
    app.root.mainloop()
