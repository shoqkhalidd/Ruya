import numpy as np 
from deep_translator import GoogleTranslator
import joblib
import keras
import openai
import os


# TTS/ STT imports
import speech_recognition as sr
import chime
from io import BytesIO
from elevenlabs import VoiceSettings, play
from elevenlabs.client import ElevenLabs

chime.theme('big-sur')

dis_model=  keras.models.load_model("Ruya/Models/distance_model.keras")
scalar = joblib.load('Ruya/Models/scaler.gz') 

def ar_eng(txt):
    translated = GoogleTranslator(source='ar', target='en').translate(txt).lower()
    pronouns= ['my', 'the','your',]
    for prounoun in pronouns:
        if prounoun in translated :
            translated = translated.split(prounoun)[-1].strip()
    return translated

def replace_misspellings(prompt):
    wrong_words = {
    'جوال': [
        'جوان', 'جواني', 'جاءال', 'جووان', 'جوالي', 'جواان', 'جوا', 'جويال', 'جوايل', 'جواول',
        'جواال', 'جال', 'جواو', 'جواي', 'جوواني', 'جوااي', 'جوااني', 'جوى', 'جواا', 'جواوان',
        'جواللي', 'جواولي', 'جواون', 'جوانا', 'جواولا', 'جواالي', 'جواوا', 'جوااذا','جواري','جوار',
        "جوا",'دوالي','دواري',
    ],
    'أمامي': [
        'عمامي', 'عمام', 'اماما', 'امامي', 'امامن', 'امامة', 'امام', 'امان', 'اماني', 'عمان',
        'عماا', 'امماي', 'اماا', 'امامن', 'اماامي', 'امامي', 'عمام', 'امامن', 'امامة', 'امام',
        'اماممي', 'امامما', 'امام', 'امامامي', 'امامم'
    ]
    }
    
    for correct, misspellings in wrong_words.items():
        for misspelled in misspellings:
            prompt = prompt.replace(misspelled, correct)
    
    return prompt

def text_to_speech_stream(text: str):
    client = ElevenLabs(
        api_key=os.environ.get('ELEVENLABS_API_KEY'),
    )  
    
    response = client.text_to_speech.convert(
        voice_id="QtP9LViXlJyZMm311fUL", 
        optimize_streaming_latency="0",
        output_format="mp3_44100_64",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0,
            use_speaker_boost=True,
        ),
    )
    audio_data = BytesIO()
    for chunk in response:
        if chunk:
            audio_data.write(chunk)
    audio_data.seek(0) 
    play(audio_data) 

def audio_transcription():
    client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    text = ''
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=48000) as source:
        r.pause_threshold = 2
        print('starting ')
        audio = r.listen(source,phrase_time_limit=8)
        try:
            wav_data = BytesIO(audio.get_wav_data())
            wav_data.name = "SpeechRecognition_audio.wav"
            transcript = client.audio.transcriptions.create(file=wav_data,language='ar', temperature=0.3 ,model='whisper-1')
        except sr.UnknownValueError:
            text_to_speech_stream("عذرًا، لم أسمعك بوضوح. هل يمكنك تكرار ما قلت؟")
    return transcript.text

# Computer Vision 
def object_location(image_width, objectX, object_width):

    image_width = image_width / 3
    object_center = objectX + (object_width / 2)
    if object_center < image_width:
        return 'left'
    elif object_center < image_width*2:
        return 'center'
    else:
        return 'right'

def which_direction(track):
    if len(track) < 2:
        return None
    start, end = np.array(track[:3])[:, 0], np.array(track[-3:])[:, 0]
    return 'right' if start.mean() < end.mean() else 'left'

def nearest_object(boxes):
    if len(boxes)==0:
        return None
    if len(boxes)==1:
        return boxes
    new_box = boxes[0]
    x1 = new_box[0]
    y1 = new_box[1]
    for box in boxes[1:]:
        x2 = box[0]
        y2 = box[1]
        if (y2 < y1) or (y2 <= y1 and x2 <x1):
            new_box = box
            x1 = x2
            y1 = y2

    return new_box

def meters_to_steps(meters):
    return int(meters * 4)

def estimate_object_distance(box):
    box = box.reshape(1, -1)
    box = scalar.transform(box)
    y_pred = dis_model.predict(box)
    return abs(y_pred[0][0])

