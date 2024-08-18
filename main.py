import Ruya.Utils as ruya
import Ruya.Agent as agent
import os

# TTS/ STT imports
import pvporcupine
import struct
import pyaudio
import chime
chime.theme('big-sur')

STOP=False

    
def open_mic():
    audio_stream = pa.open(
                 rate=porcupine.sample_rate,
                 channels=1,
                 format=pyaudio.paInt16,
                 input=True,
                 frames_per_buffer=porcupine.frame_length,
                 )
    return audio_stream
       
if __name__ == "__main__":
    
    print('here main')
    porcupine = pvporcupine.create(
    access_key=os.environ.get('PORCUPINE_API_KEY'),
    keyword_paths=["Ruya/Models/Wakeup-word/nmac.ppn"],
    model_path='Ruya/Models/Wakeup-word/porcupine_params_ar.pv'
    )

    pa = pyaudio.PyAudio()
    agent = agent.get_agent()
    audio_stream = open_mic()

    while True:
        pcm = audio_stream.read(porcupine.frame_length,exception_on_overflow=False)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        keyword_index = porcupine.process(pcm)
        
        if keyword_index >= 0:
            print(f"Detected")
            ruya.text_to_speech_stream('مرحباً، أنا مساعدتك نور، كيف يمكنني مساعدتك؟')
            audio_stream.close()
                        
            prompt = ruya.audio_transcription()
            prompt = ruya.replace_misspellings(prompt)
            print(prompt)
            chime.success()  
            
            response = agent.chat(prompt)
            ruya.text_to_speech_stream(response.response)
            audio_stream = open_mic()
            
        if STOP:
            break

        

