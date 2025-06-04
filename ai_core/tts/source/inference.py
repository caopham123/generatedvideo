from .EN.inference import OUTPUT_DIR, speak_EN
from .VI.inference import speak_VI


def speak(text, speed = 1, language = "VI", vocal = "female"):
    if language == "VI":
        speak_VI(text, speed=speed, vocal=vocal)
    else:
        speak_EN(text, speed, vocal)