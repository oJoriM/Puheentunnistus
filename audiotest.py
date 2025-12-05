"""
This script listens for audio input while a specific key is held down,
transcribes the audio using Vosk, sends the transcription to a model 
via an API, and then uses TTS to speak the response.
"""

import os
import json
import threading
import sys


if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Kuvapolut
IDLE_IMG = os.path.join(BASE_DIR, "idle.png")
HAPPY_IMG = os.path.join(BASE_DIR, "happy.png")
SAD_IMG = os.path.join(BASE_DIR, "sad.png")
YES_IMG = os.path.join(BASE_DIR, "yes.png")
NO_IMG = os.path.join(BASE_DIR, "no.png")
ANGRY_IMG = os.path.join(BASE_DIR, "angry.png")
SCARED_IMG = os.path.join(BASE_DIR, "scared.png")

# Vosk-malli
VOSK_MODEL_DIR = os.path.join(BASE_DIR, "vosk-model-en-us-0.42-gigaspeech")

import requests
import sounddevice as sd
import numpy as np
import vosk
from pynput import keyboard as pynput_keyboard
try:
    from TTS.api import TTS as CoquiTTS  
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
try:
    import pyttsx3  
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False
try:
    import simpleaudio as sa
    SIMPLEAUDIO_AVAILABLE = True
except Exception:
    sa = None
    SIMPLEAUDIO_AVAILABLE = False
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except Exception:
    sf = None
    SOUNDFILE_AVAILABLE = False
import time
try:
    import torch  
    GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    GPU_AVAILABLE = False
try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    pygame = None
    PYGAME_AVAILABLE = False



SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
KEY = os.getenv("PTT_KEY", "'")  
API_KEY = os.getenv("API_KEY", "")
VOSK_MODEL_DIR = os.getenv("VOSK_MODEL_DIR", "vosk-model-en-us-0.42-gigaspeech")
OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "http://127.0.0.1:42003")


vosk_model = None
recognizer = None
tts_model = None  
tts_engine = None  

CURRENT_PLAY_OBJ = None
PLAYBACK_MODE = None  
INTERRUPTED = False
CONVERSATION_ID = None
PTT_PRESSED = False
PTT_LISTENER = None

# Avatar systeemi
AVATAR_WINDOW = None
AVATAR_SURFACE = None
AVATAR_THREAD = None
CURRENT_AVATAR_STATE = "idle"
AVATAR_RUNNING = False
AVATAR_STATE_TIMESTAMP = time.time() 
AVATAR_STATE_DURATION = 10.0  
AVATAR_IMAGES = {}  


def stop_playback():
    """
    Stop the current playback if it's playing.
    This function is called when the 'esc' key is pressed.
    """
    global CURRENT_PLAY_OBJ, INTERRUPTED, PLAYBACK_MODE
    if PLAYBACK_MODE == 'simpleaudio':
        if CURRENT_PLAY_OBJ and CURRENT_PLAY_OBJ.is_playing():
            CURRENT_PLAY_OBJ.stop()
            INTERRUPTED = True
            print("[TTS] Playback interrupted.")
    elif PLAYBACK_MODE == 'sounddevice':
        try:
            sd.stop()
            INTERRUPTED = True
            print("[TTS] Playback interrupted.")
        except Exception:
            pass

# Funktio käynnistää äänen toiston ja kuuntelee 'esc' näppäintä keskeyttääkseen toiston
def playback_with_interrupt(audio, sample_rate):
    """
    Play the audio and listen for 'esc' key to interrupt playback.
    This function uses a separate thread to listen for the 'esc' key while the audio is playing.
    """
    global CURRENT_PLAY_OBJ, INTERRUPTED
    INTERRUPTED = False  


    if SIMPLEAUDIO_AVAILABLE:

        audio_int16 = np.int16(audio * 32767)
        CURRENT_PLAY_OBJ = sa.play_buffer(audio_int16, 1, 2, sample_rate)
        PLAYBACK_MODE = 'simpleaudio'
    else:
        CURRENT_PLAY_OBJ = None
        PLAYBACK_MODE = 'sounddevice'
        sd.play(audio, samplerate=sample_rate, blocking=False)

    def on_press(key):
        try:
            if key == pynput_keyboard.Key.esc:
                stop_playback()
                return False  
        except Exception:
            pass
        return True

    esc_listener = pynput_keyboard.Listener(on_press=on_press)
    esc_listener.daemon = True
    esc_listener.start()

    if PLAYBACK_MODE == 'simpleaudio' and CURRENT_PLAY_OBJ is not None:
        CURRENT_PLAY_OBJ.wait_done()
    else:
        sd.wait()
    return INTERRUPTED


def _is_target_ptt_key(key_obj):
    """Return True if pressed key matches configured KEY (supports 'a', '1', or 'Key.space')."""

    if isinstance(KEY, str) and KEY.startswith("Key."):
        try:
            special_name = KEY.split('.', 1)[1]
            special_key = getattr(pynput_keyboard.Key, special_name)
            return key_obj == special_key
        except Exception:
            return False

    try:
        return hasattr(key_obj, 'char') and key_obj.char == KEY
    except Exception:
        return False


def _ensure_ptt_listener():
    """Start a global keyboard listener that updates PTT_PRESSED based on KEY."""
    global PTT_LISTENER, PTT_PRESSED
    if PTT_LISTENER is not None:
        return

    def on_press(key):
        global PTT_PRESSED
        if _is_target_ptt_key(key):
            PTT_PRESSED = True

    def on_release(key):
        global PTT_PRESSED
        if _is_target_ptt_key(key):
            PTT_PRESSED = False

    PTT_LISTENER = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    PTT_LISTENER.daemon = True
    PTT_LISTENER.start()

def send_to_model(prompt_text):
    """
    Send the prompt text to the model via the Open WebUI API and return the response.
    This function constructs the API request, sends it, and processes the response.
    """
    base_url = OPENWEBUI_BASE_URL 
    url = f"{base_url}/api/chat/completions"  

    headers = {
        "Content-Type": "application/json"
    }
    if API_KEY: 
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": "gemma3:12b",
        "messages": [{"role": "user", "content": prompt_text}]
    }

    print(f"Sending to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Content: {response.text}")

        response.raise_for_status()  
        response_data = response.json()

        if response_data.get("choices"):
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            print("No valid response in API result.")
            return ""

    except requests.exceptions.RequestException as e:
        print(f"Error contacting Open WebUI: {e}")
        return ""

# Alustaa Vosk-puheentunnistusmallin ja luo tunnistimen, jos niitä ei ole vielä ladattu
def init_asr():
    """
    Initialize Vosk ASR model and recognizer lazily. Returns True on success.
    """
    global vosk_model, recognizer
    if recognizer is not None:
        return True
    if not os.path.isdir(VOSK_MODEL_DIR):
        print(f"[ASR] Vosk model directory not found: {VOSK_MODEL_DIR}")
        print("[ASR] Download a Vosk model and set VOSK_MODEL_DIR env var to its path.")
        return False
    try:
        print(f"[ASR] Loading Vosk model from '{VOSK_MODEL_DIR}'...")
        vosk_model = vosk.Model(VOSK_MODEL_DIR)
        recognizer = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
        print("[ASR] Vosk initialized.")
        return True
    except Exception as e:
        print(f"[ASR] Failed to initialize Vosk: {e}")
        return False

# Yrittää käynnistää Coqui TTS:n (GPU/CPU) ja siirtyy pyttsx3-varajärjestelmään, jos alustus epäonnistuu
def init_tts():
    """
    Initialize TTS lazily. Prefer Coqui TTS; fallback to pyttsx3 if unavailable.
    Returns True on success.
    """
    global tts_model, tts_engine
    if tts_model is not None or tts_engine is not None:
        return True
    if TTS_AVAILABLE:
        try:
            print(f"[TTS] Initializing Coqui TTS (GPU={'yes' if GPU_AVAILABLE else 'no'})...")
            tts_model = CoquiTTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=GPU_AVAILABLE)
            print("[TTS] Coqui TTS initialized.")
            return True
        except Exception as e:
            print(f"[TTS] Coqui GPU init failed or unavailable: {e}. Retrying on CPU...")
            try:
                tts_model = CoquiTTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=False)
                print("[TTS] Coqui TTS initialized on CPU.")
                return True
            except Exception as e2:
                print(f"[TTS] Failed to initialize Coqui TTS: {e2}")
                tts_model = None

    if PYTTSX3_AVAILABLE:
        try:
            print("[TTS] Initializing pyttsx3 fallback...")
            tts_engine = pyttsx3.init()

            print("[TTS] pyttsx3 initialized.")
            return True
        except Exception as e:
            print(f"[TTS] Failed to initialize pyttsx3: {e}")
            tts_engine = None
    print("[TTS] No TTS backend available. Install 'TTS' or 'pyttsx3'.")
    return False

# Muodostaa ja toistaa puhetta 
def speak(text):
    """
    Use TTS to synthesize speech from the given text.
    This function handles the TTS synthesis and playback, including error handling.
    """
    print(f"[TTS] Speaking: {text}")
    if not text:
        return
    if not init_tts():
        print("[TTS] Not initialized; skipping speech.")
        return
    if tts_model is not None:
        try:
            output = tts_model.tts(text)
            audio = np.array(output, dtype=np.float32).flatten()
        except Exception as e:
            print("[TTS] Synthesis error:", e)
            return
        sample_rate = tts_model.synthesizer.output_sample_rate
        if SOUNDFILE_AVAILABLE:
            try:
                sf.write("tts_output.wav", audio, sample_rate)
            except Exception as e:
                print(f"[TTS] Unable to write WAV: {e}")
        was_interrupted = playback_with_interrupt(audio, sample_rate)
        if was_interrupted:
            print("[TTS] Speech was interrupted. Returning to idle.")
        else:
            print("[TTS] Speech completed.")
        return
    if tts_engine is not None:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
            print("[TTS] Speech completed (pyttsx3).")
        except Exception as e:
            print(f"[TTS] pyttsx3 error: {e}")
        return
    
#Päivittää avatarin tilan ja käynnistää siihen liittyvän animaation
def animate_avatar(action):
    """
    Animate avatar based on detected speech patterns.
    This function can trigger different avatar animations based on keywords.
    """
    global CURRENT_AVATAR_STATE, AVATAR_STATE_TIMESTAMP
    CURRENT_AVATAR_STATE = action
    AVATAR_STATE_TIMESTAMP = time.time() 
    print(f"[Avatar] Animating: {action}")
    
    avatar_actions = {
        "idle": "Idle state",
        "happy": "Smiling or cheerful",
        "scared": "Surprised or scared",
        "yes": "Affirmative / agreement",
        "sad": "Looking sad or down",
        "angry": "Angry or frustrated"
    }
    
    action_desc = avatar_actions.get(action, f"Unknown action: {action}")
    print(f"[Avatar] {action_desc}")

# Lataa avatarin eri tunnetilojen kuvat tiedostoista
def load_avatar_images():
    """Load avatar images from files."""
    global AVATAR_IMAGES
    
    if not PYGAME_AVAILABLE:
        return
    
    #Kansiossa olevat kuvat
    image_map = {
        "idle": "idle.png",
        "yes": "yes.png",
        "no": "no.png",
        "happy": "happy.png",
        "angry": "angry.png",
        "sad": "sad.png",
        "scared": "scared.png"
    }
    
    for state, filename in image_map.items():
        try:
            img = pygame.image.load(filename)
            AVATAR_IMAGES[state] = img
            print(f"[Avatar] Loaded image: {filename} for state '{state}'")
        except Exception as e:
            print(f"[Avatar] Failed to load {filename}: {e}")
            AVATAR_IMAGES[state] = None

def init_avatar_window():
    """Initialize the pygame avatar window in a separate thread."""
    global AVATAR_WINDOW, AVATAR_SURFACE, AVATAR_RUNNING
    
    if not PYGAME_AVAILABLE:
        print("[Avatar] Pygame not available, skipping visual avatar.")
        return
    
    load_avatar_images()
    
    # Pyörittää avatarin pääsilmukkaa: käsittelee tapahtumat, vaihtaa tilakuvan, piirtää näytön ja ylläpitää tilapäivitysten ajoitusta
    def avatar_loop():
        global AVATAR_WINDOW, AVATAR_SURFACE, AVATAR_RUNNING, CURRENT_AVATAR_STATE, AVATAR_STATE_TIMESTAMP
        
        pygame.init()
        AVATAR_WINDOW = pygame.display.set_mode((300, 400))
        AVATAR_SURFACE = pygame.display.get_surface()
        pygame.display.set_caption("Avatar")
        AVATAR_RUNNING = True
        
        clock = pygame.time.Clock()
        animation_frame = 0
        
        #Kuvat sanojen tilan mukaan
        state_to_image = {
            "idle": "idle",
            "greeting": "happy",
            "listening": "idle",
            "happy": "happy",
            "agreement": "yes",
            "yes": "yes",
            "no": "no",
            "angry": "angry",
            "sad": "sad",
            "scared": "scared"
        }
        
        while AVATAR_RUNNING:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    AVATAR_RUNNING = False
            
            current_time = time.time()
            if current_time - AVATAR_STATE_TIMESTAMP > AVATAR_STATE_DURATION:
                if CURRENT_AVATAR_STATE not in ["speaking", "listening", "idle"]:
                    CURRENT_AVATAR_STATE = "idle"
            
            AVATAR_WINDOW.fill((240, 240, 240))
            
            image_state = state_to_image.get(CURRENT_AVATAR_STATE, "happy")
            
            if image_state and image_state in AVATAR_IMAGES and AVATAR_IMAGES[image_state] is not None:
                img = AVATAR_IMAGES[image_state]

                img_width = 250
                img_height = 250
                img_scaled = pygame.transform.scale(img, (img_width, img_height))
                
                x = (300 - img_width) // 2
                y = (400 - img_height) // 2
                
                AVATAR_WINDOW.blit(img_scaled, (x, y))
            
            font = pygame.font.Font(None, 28)
            text = font.render(CURRENT_AVATAR_STATE.upper(), True, (50, 50, 50))
            text_rect = text.get_rect(center=(150, 360))
            pygame.draw.rect(AVATAR_WINDOW, (255, 255, 255, 200), 
                           (text_rect.x - 5, text_rect.y - 2, text_rect.width + 10, text_rect.height + 4))
            AVATAR_WINDOW.blit(text, text_rect)
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
    
    AVATAR_THREAD = threading.Thread(target=avatar_loop, daemon=True)
    AVATAR_THREAD.start()
    
    time.sleep(0.5)
    print("[Avatar] Visual avatar window initialized.")

def listen_while_key_held():
    """
    Listen for audio input while the specified key is held down.
    This function captures audio, transcribes it using Vosk, and sends the transcription to the model.
    """
    asr_ok = init_asr()
    tts_ok = init_tts()
    if not asr_ok:
        print("[ASR] Not initialized; recording is disabled until the model is available.")
    if not tts_ok:
        print("[TTS] Not initialized; responses will not be spoken.")
    
    init_avatar_window()

    _ensure_ptt_listener()
    print(f"Hold '{KEY}' to speak (change with env PTT_KEY). Press Ctrl+C to exit.")
    while True:
        while not PTT_PRESSED:
            time.sleep(0.01)
        print("Recording...")
        audio_chunks = []

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=BLOCK_SIZE) as stream:
            while PTT_PRESSED:
                data, _ = stream.read(BLOCK_SIZE)
                audio_chunks.append(data)

        if len(audio_chunks) == 0:
            print("No audio captured. Try holding the key longer.")
            continue

        print("Recording stopped. Transcribing...")
        audio_data = np.concatenate(audio_chunks, axis=0)
        print("Captured audio frames:", len(audio_data))
        if recognizer is None:
            print("[ASR] Recognizer not available.")
            continue
        recognizer.Reset()
        for chunk in audio_chunks:
            recognizer.AcceptWaveform(chunk.tobytes())
        result = json.loads(recognizer.FinalResult())

        text = result.get("text", "").strip()
        if text:
            print("You said:", text)
            text_lower = text.lower()
            
            # Sanomalla nämä sanat, avatar muuttaa ilmettään. Lisätty myös muutama suomalainen sana, kirjoitettu miten ohjelma ne kuulee
            if any(word in text_lower for word in ["happy", "joy", "great", "wonderful", "excited", "elhanan", "thank you"]):
                animate_avatar("happy")
            elif any(word in text_lower for word in ["scared", "afraid", "shocked", "surprised"]):
                animate_avatar("scared")
            elif any(word in text_lower for word in ["yes", "yeah", "yep", "sure", "ok", "okay", "good luck"]):
                animate_avatar("yes")
            elif any(word in text_lower for word in ["sad", "unhappy", "depressed", "down", "i'm sorry"]):
                animate_avatar("sad")
            elif any(word in text_lower for word in ["angry", "mad", "upset", "frustrated"]):
                animate_avatar("angry")
            elif any(word in text_lower for word in ["no", "nope", "nah", "a"]):
                animate_avatar("no")
            else:
                animate_avatar("listening")
            
            response = send_to_model(text)
            
            if response:
                animate_avatar("speaking")
                print("Gemma:", response)
                speak(response)
            else:
                print("No response from model.")
        else:
            print("No speech recognized.")

if __name__ == "__main__":
    try:
        listen_while_key_held()
    except KeyboardInterrupt:
        print("\nExiting.")
        if PYGAME_AVAILABLE:
            AVATAR_RUNNING = False
            time.sleep(0.2)
