import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Load environment variables from .env
load_dotenv()
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

def recognize_once():
    # Configure speech
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "en-US"
    
    # Extend timeout to allow longer silence at start
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")  # 10 seconds

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Please speak...")

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized:", result.text)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized. Try again.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print("Canceled:", cancellation.reason)
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print("Error details:", cancellation.error_details)

if __name__ == "__main__":
    recognize_once()
