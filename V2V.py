# !pip install diffusers
# !pip install pipeline
# !pip install accelerate
# !pip install SpeechRecognition pydub


from diffusers import StableDiffusionPipeline
import torch
from IPython.display import Image, display

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


import speech_recognition as sr
from pydub import AudioSegment
import os

from google.colab import files
uploaded = files.upload()

filename = next(iter(uploaded))

if filename.endswith('.mp3'):
    sound = AudioSegment.from_mp3(filename)
    wav_filename = filename.replace('.mp3', '.wav')
    sound.export(wav_filename, format="wav")
    os.remove(filename)
    filename = wav_filename

recognizer = sr.Recognizer()

with sr.AudioFile(filename) as source:
    audio = recognizer.record(source)

try:
    text = recognizer.recognize_google(audio)
    print("Text:", text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))


prompt = "text"
image = pipe(prompt).images[0]
image.save("image.png")
display(Image("image.png"))