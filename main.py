import threading
import queue
from google.cloud import speech
import sounddevice as sd
import uuid
import requests
import time
import json
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import os


# === Setup Queues ===
audio_queue = queue.Queue()
text_queue = queue.Queue()
text_queue_for_gui = queue.Queue()
text_queue_interim = queue.Queue()
translated_text_queue = queue.Queue()


# Tambahkan flag global untuk menghentikan thread
stop_flag = False
threads = []

# Google API setup
client = speech.SpeechClient()
sample_rate = 16000
language_code = "en-US"  # Ganti ke "id-ID" kalau Bahasa Indonesia

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=sample_rate,
    language_code=language_code,  # Ganti dengan kode bahasa yang sesuai
    model="latest_long",  # Menggunakan model yang dioptimalkan untuk transkripsi berkelanjutan
    max_alternatives=1,
)

streaming_config = speech.StreamingRecognitionConfig(
    config=config,
    interim_results=True
)


def audio_callback(indata, frames, time_, status):
    if status:
        print("⚠️ Mic input error:", status)
    audio_queue.put(indata.copy())


def mic_stream_generator():
    text_queue_for_gui.put(
        "================= start listening =====================")
    print(f"stop flag is {stop_flag}")
    while not stop_flag:
        try:
            chunk = audio_queue.get(timeout=0.5)  # Timeout 0.5 detik
            yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())
        except queue.Empty:
            continue  # Lanjutkan loop jika queue kosong
    text_queue_for_gui.put(
        "================= stop listening =====================")


def transcribe_from_microphone():
    """Merekam audio dari mikrofon dan mengirimkannya ke Speech-to-Text."""
    try:
        text_queue_for_gui.put(
            "================= ▶️ started =====================")
        text_queue_interim.put(
            "================= ▶️ started =====================")
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=audio_callback):
            # Mulai streaming request
            requests = mic_stream_generator()
            responses = client.streaming_recognize(
                config=streaming_config, requests=requests)

            for response in responses:
                if stop_flag:
                    break
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue
                transcript = result.alternatives[0].transcript
                if result.is_final:
                    text_queue.put(transcript)
                    text_queue_interim.put(transcript)
                    text_queue_for_gui.put(transcript)
                    text_queue_for_gui.put(
                        "-----------------------------------------")
                    print(f"Hasil Transkripsi (Final): {transcript}")
                else:
                    print(f"Hasil Transkripsi (Interim): {transcript}")
                    text_queue_interim.put(transcript)

    except sd.PortAudioError as e:
        print(f"Error PortAudio: {e}")

    except Exception as e:
        print(f"Error Transkripsi: {e}")
        import traceback
        traceback.print_exc()  # Menampilkan stack trace untuk analisis lebih lanjut
    text_queue_for_gui.put("================= ⏹️ stopped ====================")
    text_queue_interim.put("================= ⏹️ stopped ====================")


# ==== GUI Tkinter ====


class TextDisplayGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Preprocessed and Translated Text")

        # Tombol kontrol
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        self.play_button = tk.Button(
            self.button_frame, text="Play", command=self.start_processes, font=("Arial", 12))
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(
            self.button_frame, text="Stop", command=self.stop_processes, font=("Arial", 12))
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(
            self.button_frame, text="Clear", command=self.clear_text, font=("Arial", 12))
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Panel atas untuk teks interim
        self.text_area_interim = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=80, height=4, font=("Arial", 15), bg="#ADFFF5")
        self.text_area_interim.pack(padx=10, pady=5)

        # Panel atas untuk teks dari split_text_queue
        self.text_area_original = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=80, height=12, font=("Arial", 15))
        self.text_area_original.pack(padx=10, pady=5)

        # Panel bawah untuk teks dari translated_text_queue
        self.text_area_translated = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=80, height=12, font=("Arial", 15))
        self.text_area_translated.tag_configure(
            "translated_text", foreground="#007A37", font=("Arial", 15, "bold"))
        self.text_area_translated.pack(padx=10, pady=5)

        # Jalankan fungsi update UI secara berkala
        self.update_ui()

    def start_processes(self):
        global threads, stop_flag
        stop_flag = False
        if not threads:  # Cegah memulai ulang thread yang sudah berjalan
            threads = [
                threading.Thread(target=transcribe_from_microphone),
                threading.Thread(target=translator_thread)
            ]
            for t in threads:
                t.daemon = True
                t.start()
            print("▶️ [GUI] Processes started.")

    def stop_processes(self):
        global threads, stop_flag
        stop_flag = True  # Set flag untuk menghentikan loop di thread
        # Kosongkan audio_queue
        while not audio_queue.empty():
            audio_queue.get()
        threads = []  # Kosongkan daftar thread untuk menghentikan proses
        print("⏹️ [GUI] Processes stopped.")

    def clear_text(self):
        self.text_area_original.delete(1.0, tk.END)
        self.text_area_translated.delete(1.0, tk.END)
        self.text_area_interim.delete(1.0, tk.END)
        print("🧹 [GUI] Text cleared.")

    def update_ui(self):
        # Update panel interim dengan teks dari text queui interim
        while not text_queue_interim.empty():
            sentence = text_queue_interim.get()
            self.text_area_interim.delete(1.0, tk.END)
            self.text_area_interim.insert(tk.END, sentence + "\n")
            self.text_area_interim.see(tk.END)  # scroll otomatis ke bawah

        # Update panel atas dengan teks dari split_text_queue
        while not text_queue_for_gui.empty():
            sentence = text_queue_for_gui.get()
            self.text_area_original.insert(tk.END, sentence + "\n")
            self.text_area_original.see(tk.END)  # scroll otomatis ke bawah

        # Update panel bawah dengan teks dari translated_text_queue
        while not translated_text_queue.empty():
            translated_sentence = translated_text_queue.get()
            self.text_area_translated.insert(
                tk.END, translated_sentence + "\n", "translated_text")
            self.text_area_translated.see(tk.END)  # scroll otomatis ke bawah

        self.root.after(500, self.update_ui)  # cek ulang tiap 500ms


# === Translator Thread ===


def translator_thread():
    global stop_flag
    key = os.getenv("AZURE_TRANSLATOR_KEY")
    location = "southeastasia"
    endpoint = "https://api.cognitive.microsofttranslator.com/translate"
    # lang_url =  "https://api.cognitive.microsofttranslator.com/languages?api-version=3.0"
    
    # langs_response = requests.get(lang_url).json()
    # print(langs_response)
    
    params = {
        'api-version': '3.0',
        'from': 'en-US',
        'to': ['id','jv'],
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        # location required if you're using a multi-service or regional (not global) resource.
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    translated_text_queue.put(
        "================= ▶️ started =====================")
    while not stop_flag:
        if not text_queue.empty():
            sentence = text_queue.get()
            # You can pass more than one object in body.
            body = [{'text': sentence}]
            request = requests.post(endpoint, params=params, headers=headers, json=body)
            response = request.json()
            translated = f"{response}"
            translated_text_queue.put(translated)
            translated_text_queue.put(
                "-------------------------------------------")
            print("🌍 [Translator] Translated:", translated)
            text_queue.task_done()
        else:
            time.sleep(1)
    translated_text_queue.put(
        "================= ⏹️ stopped ====================")


if __name__ == "__main__":
    # Start GUI
    root = tk.Tk()
    gui = TextDisplayGUI(root)
    root.mainloop()
