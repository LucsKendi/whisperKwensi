import whisper
import sounddevice as sd
import numpy as np
import wave
import keyboard
import os

def record_and_save_audio(directory='.', filename='audio.wav', fs=44100):
    """
    Grava áudio até que a tecla 'q' seja pressionada e salva o áudio em um arquivo.
    
    Args:
    - directory (str): Diretório onde o arquivo será salvo.
    - filename (str): Nome do arquivo a ser salvo.
    - fs (int): Taxa de amostragem do áudio.
    
    Returns:
    - str: Caminho do arquivo salvo.
    """
    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        recording.append(indata.copy())

    # Iniciar a gravação
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
        print("Gravação iniciada. Pressione 'q' para parar...")
        while True:
            if keyboard.is_pressed('q'):  # Espera pela tecla 'q'
                break
    print("Gravação parada.")

    # Concatenar todos os dados de áudio
    audio_data = np.concatenate(recording, axis=0)

    # Caminho completo do arquivo
    file_path = os.path.join(directory, filename)

    # Salvar o áudio em um arquivo WAV
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())  # Corrigir para salvar como bytes

    return file_path

# Gravar e salvar o áudio
saved_file = record_and_save_audio()

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(saved_file)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)